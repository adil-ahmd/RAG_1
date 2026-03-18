import os
import requests
import hashlib
import logging
from datetime import datetime
from urllib.parse import urlparse, unquote
from config import DATA_DIR
from domain.registry import DocumentRegistry

logger = logging.getLogger(__name__)

class DocumentDownloader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def download_if_needed(self, url: str, registry: DocumentRegistry) -> str:
        """
        Downloads a PDF if it's new or modified based on ETag/Last-Modified HTTP headers.
        Returns the local filepath of the downloaded (or existing) file.
        Returns None if download failed.
        """
        try:
            # Check headers
            head_response = requests.head(url, timeout=10)
            etag = head_response.headers.get("ETag")
            last_modified = head_response.headers.get("Last-Modified")
            
            # Identify file
            filename = unquote(os.path.basename(urlparse(url).path))
            if not filename.endswith('.pdf'):
                filename = f"doc_{hashlib.md5(url.encode()).hexdigest()}.pdf"

            # Check network-level registry cache to skip download if unchanged
            registry_entry = registry._registry.get(url)  # use URL as the ID for network tracking
            
            needs_download = True
            if registry_entry:
                if etag and registry_entry.get("etag") == etag:
                    needs_download = False
                elif last_modified and registry_entry.get("last_modified") == last_modified:
                    needs_download = False
            
            if not needs_download:
                logger.info(f"Skipping download {url} - No network changes detected.")
                return registry_entry.get("filepath")

            # Download
            logger.info(f"Downloading {url}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Ensure PDF signature
            if not response.content.startswith(b"%PDF"):
                logger.warning(f"Skipping {url}: Not a valid PDF file.")
                return None

            # Keep Version History: Timestamp the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(self.data_dir, final_filename)

            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Update the registry with network metadata so we can skip next time
            network_metadata = {
                "filepath": filepath,
                "etag": etag,
                "last_modified": last_modified,
                "download_date": timestamp
            }
            # We are storing network data under the URL key in the same registry for simplicity
            registry._registry[url] = network_metadata
            registry._save()
            
            return filepath

        except Exception as e:
            logger.error(f"Failed to process download for {url}: {e}")
            return None
