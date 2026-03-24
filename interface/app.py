import sys
import os
import streamlit as st
from infrastructure.vector_index_manager import VectorIndexManager
from application.query_service import QueryService
from config import INDEX_DIR, LLM_MODEL, LLM_PROVIDER, EMBEDDING_MODEL
import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Page Config
st.set_page_config(page_title="ZATCA Tax Assistant")

st.title("ZATCA Tax Assistant")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if "qa_service" in st.session_state and st.session_state.qa_service:
            st.session_state.qa_service.chat_history = []
        st.rerun()

def init_qa_service():
    if "qa_service" not in st.session_state:
        with st.spinner("Loading Knowledge Base..."):
            index_manager = VectorIndexManager(index_dir=INDEX_DIR, embedding_model_name=EMBEDDING_MODEL)
            retriever = index_manager.get_retriever(
    search_type="mmr",               # Max Marginal Relevance - diverse results
    search_kwargs={"k": 5, "fetch_k": 20}  # Fetch 5 best from top 20
)
            if not retriever:
                st.session_state.qa_service = None
                return
                
            if LLM_PROVIDER == "gemini":
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL,
                    temperature=0,
                    convert_system_message_to_human=True
                )
            else:
                llm = None
            
            st.session_state.qa_service = QueryService(retriever=retriever, llm=llm)

init_qa_service()

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about tax policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if st.session_state.qa_service:
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.qa_service.ask(prompt)
                    answer = result['answer']
                    sources = result['sources']
                    
                    # Format Sources
                    sources_text = "\n\n**Sources:**\n"
                    for src in sources:
                        sources_text += f"- *{src}*\n"
                        
                    full_response = answer + sources_text if sources else answer
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                message_placeholder.error(error_msg)
        else:
            message_placeholder.error("Knowledge base not loaded. Please ensure 'data/' folder exists and run Ingest.")
