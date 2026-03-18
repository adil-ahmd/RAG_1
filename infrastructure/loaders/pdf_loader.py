import logging
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from domain.models import Document

logger = logging.getLogger(__name__)

class PDFLoader:
    def load(self, source_path: str) -> list[Document]:
        """
        Loads a PDF and returns a list of Domain Document objects.
        Returns one Document combined for the whole file to easily hash and chunk.
        """
        try:
            loader = PyPDFLoader(source_path)
            langchain_docs = loader.load()
            
            if not langchain_docs:
                return []
                
            full_content = "\n\n".join(doc.page_content for doc in langchain_docs)
            content_hash = hashlib.sha256(full_content.encode('utf-8')).hexdigest()
            
            metadata = {
                "source": source_path
            }
            if langchain_docs and langchain_docs[0].metadata:
                metadata.update(langchain_docs[0].metadata)
                
            doc = Document(
                id=source_path,
                content=full_content,
                metadata=metadata,
                hash=content_hash
            )
            return [doc]
        except Exception as e:
            logger.error(f"Failed to load PDF {source_path}: {e}")
            return []
