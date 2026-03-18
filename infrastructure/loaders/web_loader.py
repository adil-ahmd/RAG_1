import logging
from domain.models import Document

logger = logging.getLogger(__name__)

class WebLoader:
    def load(self, source_url: str) -> list[Document]:
        """
        Placeholder for web loading logic.
        """
        logger.warning("WebLoader not fully implemented.")
        return []
