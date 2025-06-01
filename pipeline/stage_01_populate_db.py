import os
import traceback
import shutil
from dotenv import load_dotenv
from utils.config import CONFIG
from utils.logger import setup_logger
import requests
from typing import List
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.docstore.document import Document


load_dotenv(dotenv_path=".env.local")
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
URLS = CONFIG["URLS"]
LOG_PATH = CONFIG["LOG_PATH"]
CHROMA_DB_PATH = CONFIG["CHROMA_DB_PATH"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db.log")


def scrape_with_firecrawl(url, logger):
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "url": url,
        # "mode": "scrape", # deprecated
        "formats": ["markdown"], # Request markdown directly
        "onlyMainContent": True,
        "removeBase64Images": True,
        "blockAds": True,
        "proxy": "basic",
        "timeout": 30000
    }
    response = requests.post(endpoint, headers=headers, json=body)

    if response.status_code == 403:
        logger.warning(f"[- BLOCKED] {url} - This site is not supported by Firecrawl.")
        return None
    if response.status_code != 200:
        logger.warning(f"[- FAIL] {url} - Status: {response.status_code} - {response.text}")
        return None
    
    result = response.json()
    content = result.get("data", {}).get("markdown", "").strip()

    if not content:
        logger.warning(f"[!] No content extracted from {url}")
        return None

    return Document(page_content=content, metadata={"url": url})

def scrape_documents(urls: List[str], logger) -> List[Document]:
    docs = []
    for url in urls:
        try:
            doc = scrape_with_firecrawl(url, logger)
            if doc and doc.page_content:
                docs.append(doc)
                logger.info(f"[+] Scraped: {url}")
            logger.info(f"[+] Content length: {len(doc.page_content) if doc else 0}")
        except Exception as e:
            logger.error(f"[-] Failed: {url} - {e}")
            logger.debug(traceback.format_exc())
    return docs

def flatten_documents(docs: List, logger) -> List[Document]:
    try:
        if docs and isinstance(docs[0], list):
            docs_list = [item for sublist in docs for item in sublist]
            logger.info(f"[+] Flattened docs count: {len(docs_list)}")
        else:
            docs_list = docs
            logger.info(f"[+] Docs already flat, count: {len(docs_list)}")
    except Exception as e:
        logger.error(f"[-] Error flattening docs list: {e}")
        logger.debug(traceback.format_exc())
        docs_list = docs  # fallback
    return docs_list

def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int, logger) -> List[Document]:
    try:
        logger.info("[+] Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"[+] Total chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"[-] Error splitting documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def filter_metadata(docs_split: List[Document], logger) -> List[Document]:
    filtered_docs = []
    for i, doc in enumerate(docs_split):
        try:
            if isinstance(doc, Document) and hasattr(doc, 'metadata'):
                clean_metadata = {
                    k: v for k, v in doc.metadata.items()
                    if isinstance(v, (str, int, float, bool))
                }
                filtered_doc = Document(
                    page_content=doc.page_content,
                    metadata=clean_metadata
                )
                filtered_docs.append(filtered_doc)
            else:
                logger.warning(f"[!] Skipping doc at index {i}: Invalid type or missing metadata")
        except Exception as e:
            logger.error(f"[-] Error processing doc at index {i}: {e}")
            logger.debug(traceback.format_exc())
    logger.info(f"[+] Filtered docs count: {len(filtered_docs)}")
    return filtered_docs

def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
    for idx, chunk in enumerate(chunks):
        url = chunk.metadata.get("url", "unknown_url")
        chunk.metadata["chunk_id"] = f"{url}:{idx}"
    return chunks

def process_in_batches(documents, batch_size, ingest_fn, logger):
    total = len(documents)
    logger.info(f"Processing {total} documents in batches of {batch_size}...")
    for i in tqdm(range(0, total, batch_size), desc="Ingesting batches"):
        batch = documents[i:i + batch_size]
        try:
            ingest_fn(batch)
        except Exception as e:
            logger.error(f"Failed to ingest batch {i // batch_size}: {e}")
            logger.debug(traceback.format_exc())

def save_to_chroma_db(chunks: List[Document], logger):
    try:
        logger.info("Saving chunks to Chroma DB...")
        db = Chroma(
            embedding_function=GPT4AllEmbeddings(),
            persist_directory=CHROMA_DB_PATH
        )
        chunks = assign_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])

        new_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing_ids]
        seen_ids = set()
        unique_chunks = []
        for chunk in new_chunks:
            cid = chunk.metadata["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)

        if unique_chunks:
            process_in_batches(
                documents=unique_chunks,
                batch_size=BATCH_SIZE,
                ingest_fn=lambda batch: db.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch]),
                logger=logger
            )
        else:
            logger.info("No new unique chunks to add.")
        logger.info("Chunks saved successfully.")
    except Exception as e:
        logger.error(f"Error saving to DB: {e}")
        logger.debug(traceback.format_exc())

def clear_database():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

def run_populate_db(urls, reset=False):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info("Starting DB population pipeline...")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("Clearing the database...")
            clear_database()
        
        scraped = scrape_documents(urls, logger)
        if not scraped:
            logger.warning("No documents scraped. Exiting.")
            return
        flat_docs = flatten_documents(scraped, logger)
        split = split_documents(flat_docs, CHUNK_SIZE, CHUNK_OVERLAP, logger)
        filtered = filter_metadata(split, logger)
        save_to_chroma_db(filtered, logger)
        logger.info("DB population pipeline completed.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_populate_db(URLS)