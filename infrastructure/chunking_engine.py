import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LC_Document
from domain.models import Document

logger = logging.getLogger(__name__)

class ChunkingEngine:
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )

    def chunk(self, document: Document):
        """
        Transforms Domain Document into a generator of Langchain Documents (Chunks).
        No metadata injection into content! Yields chunks lazily.
        """
        try:
            chunks = self.text_splitter.split_text(document.content)
            
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = document.metadata.copy()
                chunk_metadata["chunk_id"] = f"{document.id}::chunk_{i}"
                chunk_metadata["doc_id"] = document.id
                
                yield LC_Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
            
        except Exception as e:
            logger.error(f"Failed to chunk document {document.id}: {e}")
