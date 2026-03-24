import logging
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain

logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self, retriever, llm):
        """
        QueryService orchestrates the Retrieval -> LLM flow.
        Uses StuffDocumentsChain directly instead of RetrievalQA so that
        extra input variables like chat_history are fully supported.
        Chat memory uses token-aware trimming to stay within LLM context limits.
        """
        self.retriever = retriever
        self.llm = llm
        self.chat_history = []

        # Max characters allowed in chat history passed to LLM
        # ~4 chars per token, keeping last ~800 tokens worth of history
        self.MAX_HISTORY_CHARS = 3200

        template = """You are an expert ZATCA (Zakat, Tax and Customs Authority) tax assistant for Saudi Arabia.

Behavior rules:
1. If the user message is a greeting or casual small talk, respond politely and briefly introduce yourself.
2. If the user asks a tax-related question, answer using the context provided below.
3. Be as detailed and helpful as possible. Include:
   - Specific rates, thresholds, or deadlines if mentioned in the context
   - Practical implications for businesses or individuals
   - Any exceptions or special cases found in the context
4. If the context is partially relevant, use what is available and clearly state which parts you are inferring.
5. If the answer is completely absent from the context, say: "This specific topic is not covered in the available ZATCA documents. Please refer to zatca.gov.sa for official guidance."
6. Always respond in a clear, structured format using bullet points or numbered lists where appropriate.
7. If the question involves a calculation, show the formula and an example.

Context (use this as your sole source of information):
{context}

Conversation History (for reference resolution only):
{chat_history}
⚠️ IMPORTANT: Use the conversation history ONLY to understand what the user is referring to
(e.g. resolving pronouns like "that", "it", "those", or "the above").
Do NOT use previous answers as a source of information.
Do NOT repeat, summarize, or build upon previous answers.
Your answer must be based SOLELY on the Context provided above.

Question:
{question}

Detailed Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        self.combine_docs_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )

    def _format_history(self) -> str:
        """
        Formats chat history into a string, trimming from the oldest messages
        if total character count exceeds MAX_HISTORY_CHARS.
        This prevents context window overflow on long conversations.
        """
        if not self.chat_history:
            return ""

        lines = [f"{msg['role']}: {msg['content']}" for msg in self.chat_history]

        # Trim oldest messages until within limit
        while lines:
            combined = "\n".join(lines)
            if len(combined) <= self.MAX_HISTORY_CHARS:
                return combined
            lines.pop(0)  # Remove oldest message

        return ""

    def _update_history(self, question: str):
        """
        Stores only the user question in history, not the assistant answer.
        History is used solely for reference resolution (e.g. "that", "it", "those").
        Storing answers would cause the LLM to reuse them as information sources.
        """
        self.chat_history.append({"role": "User", "content": question})

    def ask(self, question: str, use_memory: bool = True) -> dict:
        """
        Ask a question and get a dictionary containing the answer and sources.
        """
        try:
            formatted_history = self._format_history() if use_memory else ""

            # Retrieve relevant documents
            docs = self.retriever.invoke(question)

            # Run LLM with docs + question + history
            answer = self.combine_docs_chain.run(
                input_documents=docs,
                question=question,
                chat_history=formatted_history
            )

            # Deduplicate sources
            sources = []
            seen = set()
            for doc in docs:
                src = doc.metadata.get('source', 'Unknown')
                if src not in seen:
                    sources.append(src)
                    seen.add(src)

            if use_memory:
                self._update_history(question)

            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                "answer": f"Error querying knowledge base: {str(e)}",
                "sources": []
            }
