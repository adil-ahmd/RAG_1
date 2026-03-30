import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LC_Document

logger = logging.getLogger(__name__)

class VectorIndexManager:
    def __init__(self, index_dir: str, embedding_model_name: str):
        self.index_dir = index_dir
        self.embedding_model_name = embedding_model_name
        self.embeddings = None 
        self.vectorstore = None 
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization - only load when actually needed"""
        if self._initialized:
            return
            
        logger.info("Loading embeddings and index (first use)...")
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_index()
        self._initialized = True
        logger.info("Index manager fully initialized.")

    def _load_embeddings(self):
        logger.info(f"Loading embeddings model: {self.embedding_model_name}")
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model_name, 
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 32}
        )

    def _load_index(self):
        """Load FAISS index from disk"""
        faiss_file = os.path.join(self.index_dir, "index.faiss")
        if not os.path.exists(faiss_file):
            logger.warning(f"Index file not found at {faiss_file}. Will create new one.")
            return None
            
        logger.info(f"Loading FAISS index from {self.index_dir}...")
        return FAISS.load_local(
            self.index_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )

    def load_index(self):
        """Public method for backward compatibility"""
        self._ensure_initialized()
        return self.vectorstore

    def replace(self, doc_id: str, chunks: list[LC_Document]):
        """
        Deletes old chunks associated with the doc_id and adds the new ones.
        """
        self._ensure_initialized()  
        
        if not chunks:
            return

        if self.vectorstore is None:
            logger.info(f"Creating new index with {len(chunks)} chunks for {doc_id}...")
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self._delete_by_doc_id(doc_id)
            self.vectorstore.add_documents(chunks)
            
        self.save()

    def _delete_by_doc_id(self, doc_id: str):
        """
        Deletes documents matching the specified doc_id.
        """
        if not self.vectorstore:
            return
            
        ids_to_delete = []
        for docstore_id, doc in self.vectorstore.docstore._dict.items():
            if doc.metadata.get("doc_id") == doc_id:
                ids_to_delete.append(docstore_id)
                
        if ids_to_delete:
            logger.info(f"Deleting {len(ids_to_delete)} stale chunks for doc_id {doc_id}")
            self.vectorstore.delete(ids_to_delete)

    def save(self):
        if self.vectorstore:
            self.vectorstore.save_local(self.index_dir)
            logger.info(f"Index updated and saved to {self.index_dir}")

    def get_retriever(self, **kwargs):
        """
        Exposes the FAISS retriever for the query side.
        """
        self._ensure_initialized()  
        
        if self.vectorstore:
            return self.vectorstore.as_retriever(**kwargs)
        return None