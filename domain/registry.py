import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DocumentRegistry:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self._registry: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode {self.storage_path}. Starting empty.")
                    return {}
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self._registry, f, indent=4)

    def has_changed(self, doc_id: str, new_hash: str) -> bool:
        """
        Check if the document has changed based on its hash.
        Returns True if it's new or the hash differs.
        """
        if doc_id not in self._registry:
            return True
        return self._registry[doc_id].get("hash") != new_hash

    def update(self, doc_id: str, doc_hash: str):
        """
        Update the registry with the latest hash.
        """
        if doc_id not in self._registry:
            self._registry[doc_id] = {}
        self._registry[doc_id]["hash"] = doc_hash
        self._save()
