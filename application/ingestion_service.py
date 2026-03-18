import logging
import os
from domain.models import Document

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self, downloader, loader, chunker, index_manager, registry):
        self.downloader = downloader
        self.loader = loader
        self.chunker = chunker
        self.index_manager = index_manager
        self.registry = registry

    def ingest_url(self, url: str):
        """
        Downloads a URL if needed, then ingests the resulting local file.
        """
        logger.info(f"Checking URL: {url}")
        filepath = self.downloader.download_if_needed(url, self.registry)
        
        if filepath:
            self.ingest(filepath)

    def ingest(self, source_path: str):
        """
        Orchestrates the ingestion pipeline for a single local source file.
        """
        if not os.path.exists(source_path):
            logger.error(f"Source path {source_path} does not exist.")
            return

        logger.info(f"Ingesting source: {source_path}")
        documents = self.loader.load(source_path)
        
        for document in documents:
            if not self.registry.has_changed(document.id, document.hash):
                logger.info(f"Skipping {document.id} - No changes detected.")
                continue
            
            logger.info(f"Processing and chunking document {document.id}...")
            chunks = list(self.chunker.chunk(document))
            
            self.index_manager.replace(document.id, chunks)
            
            self.registry.update(document.id, document.hash)
            logger.info(f"Successfully ingested {document.id}.")
