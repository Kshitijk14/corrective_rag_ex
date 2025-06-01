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
from langchain.docstore.document import Document
from utils.get_llm_func import embedding_func


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
            logger.info(f"[->] Content length: {len(doc.page_content) if doc else 0}")
        except Exception as e:
            logger.error(f"Failed: {url} - {e}")
            logger.debug(traceback.format_exc())
    return docs

def flatten_documents(docs: List, logger) -> List[Document]:
    try:
        if docs and isinstance(docs[0], list):
            docs_list = [item for sublist in docs for item in sublist]
            logger.info(f"[Part 01(a)] Flattened docs count: {len(docs_list)}")
        else:
            docs_list = docs
            logger.info(f"[Part 01(b)] Docs already flat, count: {len(docs_list)}")
    except Exception as e:
        logger.error(f"Error flattening docs list: {e}")
        logger.debug(traceback.format_exc())
        docs_list = docs  # fallback
    return docs_list

def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int, logger) -> List[Document]:
    try:
        logger.info("[Part 02] Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"[Part 03] Total chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
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
                logger.warning(f"[Part 04] Skipping doc at index {i}: Invalid type or missing metadata")
        except Exception as e:
            logger.error(f"Error processing doc at index {i}: {e}")
            logger.debug(traceback.format_exc())
    logger.info(f"[Part 05] Filtered docs count: {len(filtered_docs)}")
    return filtered_docs

def assign_chunk_ids(chunks: List[Document]) -> List[Document]:
    for idx, chunk in enumerate(chunks):
        url = chunk.metadata.get("url", "unknown_url")
        chunk.metadata["chunk_id"] = f"{url}:{idx}"
    return chunks

def process_in_batches(documents, batch_size, ingest_fn, logger):
    total = len(documents)
    logger.info(f"[Part 06] Processing {total} documents in batches of {batch_size}...")
    for i in tqdm(range(0, total, batch_size), desc="Ingesting batches"):
        batch = documents[i:i + batch_size]
        try:
            ingest_fn(batch)
        except Exception as e:
            logger.error(f"Failed to ingest batch {i // batch_size}: {e}")
            logger.debug(traceback.format_exc())

def save_to_chroma_db(chunks: List[Document], logger):
    try:
        logger.info("[Part 07] Saving chunks to Chroma DB...")
        db = Chroma(
            embedding_function=embedding_func(),
            persist_directory=CHROMA_DB_PATH
        )
        logger.info(f"[Part 08] Loading existing DB from path: {CHROMA_DB_PATH}")
        
        chunks = assign_chunk_ids(chunks)
        logger.info(f"[Part 09] Total chunks to add: {len(chunks)}")
        
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        logger.info(f"[Part 10] Existing DB items count: {len(existing_ids)}")

        new_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_id"] not in existing_ids]
        logger.info(f"[Part 11] New chunks to add: {len(new_chunks)}")
        
        seen_ids = set()
        unique_chunks = []
        for chunk in new_chunks:
            cid = chunk.metadata["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_chunks.append(chunk)
        logger.info(f"[Part 12] Unique chunks to add: {len(unique_chunks)}")

        if unique_chunks:
            logger.info("[Part 13(a)] Ingesting new unique chunks to DB in batches...")
            process_in_batches(
                documents=unique_chunks,
                batch_size=BATCH_SIZE,
                ingest_fn=lambda batch: db.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch]),
                logger=logger
            )
        else:
            logger.info("[Part 13(b)] No new unique chunks to add.")
        
        logger.info("Chunks saved successfully.")
    except Exception as e:
        logger.error(f"Error saving to Chroma DB: {e}")
        logger.debug(traceback.format_exc())

def clear_database():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

def run_populate_db(urls, reset=False):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("++++++++Starting DB population pipeline...")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("[RESET DB] Clearing the database...")
            clear_database()
        
        scraped = scrape_documents(urls, logger)
        if not scraped:
            logger.warning("No documents scraped. Exiting.")
            return
        
        flat_docs = flatten_documents(scraped, logger)
        split = split_documents(flat_docs, CHUNK_SIZE, CHUNK_OVERLAP, logger)
        filtered = filter_metadata(split, logger)
        save_to_chroma_db(filtered, logger)
        
        logger.info("++++++++DB population pipeline completed.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_populate_db(URLS)