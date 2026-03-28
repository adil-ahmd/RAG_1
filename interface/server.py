import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import warnings

# Suppress Pydantic V1 / Python 3.14 deprecation warning coming from langchain_core
warnings.filterwarnings("ignore", message=".*Core Pydantic V1 functionality isn't compatible with Python 3.14.*")
warnings.filterwarnings("ignore")

try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError:
    class Context:
        pass

    class FastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator

from config import INDEX_DIR, LLM_PROVIDER, LLM_MODEL, EMBEDDING_MODEL

SERVER_NAME = "ZATCA RAG"

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(__name__)

mcp = FastMCP(SERVER_NAME)
shared_index_manager = None
query_service_instance = None
ingestion_service_instance = None

def get_shared_index_manager():
    """Lazy load - only when first tool is called"""
    global shared_index_manager
    if not shared_index_manager:
        logger.info("Initializing index manager (first call)...")
        from infrastructure.vector_index_manager import VectorIndexManager
        shared_index_manager = VectorIndexManager(
            index_dir=INDEX_DIR,
            embedding_model_name=EMBEDDING_MODEL
        )
        logger.info("Index manager ready.")
    return shared_index_manager

def get_query_service():
    """Lazy load - only when query tool is called"""
    global query_service_instance
    if query_service_instance:
        return query_service_instance

    logger.info("Initializing query service (first call)...")
    index_manager = get_shared_index_manager()
    retriever = index_manager.get_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    if not retriever:
        logger.error("No retriever available - index may be empty")
        return None

    if LLM_PROVIDER == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,
            convert_system_message_to_human=True
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. Only 'gemini' is supported.")

    from application.query_service import QueryService
    query_service_instance = QueryService(retriever=retriever, llm=llm)
    logger.info("Query service ready.")
    return query_service_instance

def get_ingestion_service():
    """Lazy load - only when ingestion tool is called"""
    global ingestion_service_instance
    if ingestion_service_instance:
        return ingestion_service_instance

    logger.info("Initializing ingestion service (first call)...")
    from infrastructure.loaders.pdf_loader import PDFLoader
    from infrastructure.chunking_engine import ChunkingEngine
    from domain.registry import DocumentRegistry
    from application.ingestion_service import IngestionService
    from config import METADATA_FILE

    try:
        from infrastructure.loaders.downloader import DocumentDownloader
        downloader = DocumentDownloader()
    except ImportError:
        logger.warning("DocumentDownloader not found, using None")
        downloader = None

    loader = PDFLoader()
    chunker = ChunkingEngine()
    index_manager = get_shared_index_manager()
    registry = DocumentRegistry(storage_path=METADATA_FILE)

    ingestion_service_instance = IngestionService(
        downloader=downloader,
        loader=loader,
        chunker=chunker,
        index_manager=index_manager,
        registry=registry
    )
    logger.info("Ingestion service ready.")
    return ingestion_service_instance

@mcp.tool()
def query_zatca_knowledge(query: str, ctx: Context = None) -> str:
    """
    Queries the ZATCA knowledge base.

    ⚠️ IMPORTANT: First call takes 15-30 seconds to load the index.
    Subsequent calls are instant.

    Returns the answer with source citations.
    """
    try:
        logger.info(f"Query received: {query[:50]}...")
        if ctx:
            ctx.info(f"Query received: {query}")
            ctx.report_progress(1, 100)
            ctx.info("Initializing knowledge base... This may take up to 30 seconds on the first run.")

        service = get_query_service()

        if ctx:
            ctx.report_progress(50, 100)

        if not service:
            return "Error: Knowledge base not loaded. Run check_for_updates first."

        logger.info("Executing query...")
        if ctx:
            ctx.info("Executing query against LLM...")
            ctx.report_progress(75, 100)

        result = service.ask(query, use_memory=False)
        answer = result['answer']
        sources = result['sources']

        if ctx:
            ctx.report_progress(100, 100)

        if sources:
            return f"Answer: {answer}\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
        return f"Answer: {answer}"

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        if ctx:
            ctx.error(f"Query failed: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def calculate_vat(amount: float, is_export: bool = False) -> str:
    """
    Calculates the VAT for a given amount.
    Standard rate is 15%. Exports are 0%.

    This is a pure calculation - no knowledge base needed.
    """
    rate = 0.0 if is_export else 0.15
    vat_amount = amount * rate
    total = amount + vat_amount

    return (f"Base Amount: {amount:.2f} SAR\n"
            f"VAT Rate: {rate*100}%\n"
            f"VAT Amount: {vat_amount:.2f} SAR\n"
            f"Total: {total:.2f} SAR")

@mcp.tool()
def run_crawler(ctx: Context = None) -> str:
    """
    Launches the ZATCA web crawler as a background process and returns immediately.

    The crawler will run in the background and save results to zatca_documents.json.
    After it completes, call check_for_updates() to ingest the new documents.
    """
    try:
        from config import BASE_DIR
        import subprocess

        spider_path = os.path.join(BASE_DIR, "crawler", "zatca_selenium_scraper.py")
        output_file = os.path.join(BASE_DIR, "crawler", "zatca_documents.json")
        log_file_path = os.path.join(BASE_DIR, "crawler", "crawler.log")

        if not os.path.exists(spider_path):
            return f"Error: Crawler script not found at {spider_path}"

        log_file = open(log_file_path, "w")

        process = subprocess.Popen(
            [sys.executable, spider_path],
            stdout=log_file,
            stderr=log_file,
            close_fds=True,
            start_new_session=True
        )

        if ctx:
            ctx.info(f"Crawler launched with PID {process.pid}")

        return (
            f"✅ Crawler started in background (PID: {process.pid})\n\n"
            f"📄 Output will be saved to: {output_file}\n"
            f"📋 Logs available at: {log_file_path}\n\n"
            f"⏳ Wait 2-5 minutes for it to complete, then call check_for_updates() to ingest results."
        )

    except Exception as e:
        logger.error(f"Failed to launch crawler: {e}", exc_info=True)
        if ctx:
            ctx.error(f"Failed to launch crawler: {e}")
        return f"Error launching crawler: {str(e)}"

@mcp.tool()
def read_project_file(filepath: str) -> str:
    """Read a file from the project directory."""
    from config import BASE_DIR
    full_path = os.path.join(BASE_DIR, filepath)
    if not os.path.exists(full_path):
        return f"File not found: {full_path}"
    with open(full_path, 'r') as f:
        return f.read()

@mcp.tool()
def check_for_updates(ctx: Context = None) -> str:
    """
    Triggers the ingestion pipeline:
    1. Runs ZATCA web crawler to find PDFs
    2. Downloads and processes new documents
    3. Updates the knowledge base index

    This may take several minutes on first run.
    """
    try:
        if ctx:
            ctx.info("Initializing ingestion service...")
            ctx.report_progress(1, 100)

        service = get_ingestion_service()
        from config import BASE_DIR
        import json

        output_file = os.path.join(BASE_DIR, "crawler", "zatca_documents.json")

        if not os.path.exists(output_file):
            return (
                "No crawler output found. Please run the crawler manually first:\n"
                f"  python3 {os.path.join(BASE_DIR, 'crawler', 'zatca_selenium_scraper.py')}"
            )

        if ctx:
            ctx.info("Loading crawler results...")
            ctx.report_progress(10, 100)

        with open(output_file, 'r') as f:
            documents = json.load(f)

        if not documents:
            return "Crawler output file is empty. Re-run the crawler."

        added_count = 0
        errors = []
        total_docs = len(documents)

        for i, doc in enumerate(documents):
            url = doc.get('url')
            if url:
                if ctx:
                    curr_prog = 10 + int((i / total_docs) * 85)
                    ctx.report_progress(curr_prog, 100)
                    ctx.info(f"Processing {i+1}/{total_docs}: {url.split('/')[-1]}")
                try:
                    service.ingest_url(url)
                    added_count += 1
                except Exception as e:
                    errors.append(f"{url}: {str(e)[:100]}")
                    logger.error(f"Failed to ingest {url}: {e}")

        if ctx:
            ctx.report_progress(100, 100)
            ctx.info("Update complete.")

        result_msg = f"Successfully processed {added_count}/{total_docs} documents."
        if errors:
            result_msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])

        return result_msg

    except Exception as e:
        logger.error(f"Update failed: {e}", exc_info=True)
        if ctx:
            ctx.error(f"Update failed: {e}")
        return f"Error checking for updates: {str(e)}"

if __name__ == "__main__":
    if sys.stdin.isatty():
        logger.warning("="*60)
        logger.warning("ERROR: This is an MCP Server.")
        logger.warning("It speaks JSON-RPC over stdio, not human text.")
        logger.warning("-" * 60)
        logger.warning("To inspect: npx @modelcontextprotocol/inspector mcp run server.py")
        logger.warning("To use: Configure in Claude Desktop settings")
        logger.warning("="*60)
        sys.exit(1)

    logger.info("ZATCA RAG MCP Server starting...")
    logger.info("Tools will initialize on first use (lazy loading)")
    mcp.run()
