import logging
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

logger = logging.getLogger(__name__)

class CustomRetrievalQA(RetrievalQA):
    def _call(self, inputs, run_manager=None):
        question = inputs["query"]
        chat_history = inputs.get("chat_history", "")
        accepts_run_manager = (
            "run_manager" in self.combine_documents_chain.__call__.__code__.co_varnames
        )
        docs = self._get_docs(question, run_manager=run_manager)
        answer = self.combine_documents_chain.run(
            input_documents=docs, 
            question=question, 
            chat_history=chat_history,
            callbacks=run_manager.get_child() if run_manager else None
        )
        result = {"result": answer}
        if self.return_source_documents:
            result["source_documents"] = docs
        return result

class QueryService:
    def __init__(self, retriever, llm):
        """
        QueryService orchestrates the Retrieval -> LLM flow.
        No FAISS instantiation here.
        """
        self.retriever = retriever
        self.llm = llm
        self.chat_history = []

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

Context:
{context}

Chat History (last 5 messages):
{chat_history}

Question:
{question}

Detailed Answer:"""

        qa_chain_prompt = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )

        self.qa_chain = CustomRetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt},
            verbose=False
        )

    def ask(self, question: str, use_memory: bool = True) -> dict:
        """
        Ask a question and get a dictionary containing the answer and sources.
        """
        try:
            # Format chat history
            formatted_history = ""
            if use_memory and self.chat_history:
                formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history])
            
            result = self.qa_chain.invoke({
                "query": question,
                "chat_history": formatted_history
            })

            answer = result['result']
            sources = []
            seen = set()
            for doc in result['source_documents']:
                src = doc.metadata.get('source', 'Unknown')
                if src not in seen:
                    sources.append(src)
                    seen.add(src)

            if use_memory:
                self.chat_history.append({"role": "User", "content": question})
                self.chat_history.append({"role": "Assistant", "content": answer})
                # Keep only the last 5 turns (10 messages)
                if len(self.chat_history) > 10:
                    self.chat_history = self.chat_history[-10:]

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