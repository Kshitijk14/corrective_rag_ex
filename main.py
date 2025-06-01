import os
import argparse
import traceback
from utils.config import CONFIG
from utils.logger import setup_logger
from pipeline.stage_01_populate_db import run_populate_db
from pipeline.stage_02_grade_retrieved_docs import run_grade_retrieved_docs

URLS = CONFIG["URLS"]
LOG_PATH = CONFIG["LOG_PATH"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "main.log")


def main():
    # Create CLI.
    parser = argparse.ArgumentParser(description="Main Pipeline")
    parser.add_argument("--scrape", action="store_true", help="Scrape URLs and save content")
    parser.add_argument("--reset", action="store_true", help="Reset Chroma DB before population")
    args = parser.parse_args()
    
    logger = setup_logger("main_logger", LOG_FILE)
    
    try:
        logger.info(" ")
        logger.info("////--//--//----STARTING [PIPELINE 01]: POPULATE DB----//--//--////")
        run_populate_db(URLS, scrape=args.scrape, reset=args.reset)
        logger.info("////--//--//----FINISHED [PIPELINE 01]: POPULATE DB----//--//--////")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error in [PIPELINE 01]: POPULATE DB: {e}")
        logger.debug(traceback.format_exc())
        return
    
    try:
        logger.info(" ")
        logger.info("////--//--//----STARTING [PIPELINE 02]: QUERY RAG & GRADE RETRIEVED DOCS----//--//--////")
        run_grade_retrieved_docs()
        logger.info("////--//--//----FINISHED [PIPELINE 02]: QUERY RAG & GRADE RETRIEVED DOCS----//--//--////")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error in [PIPELINE 02]: POPULATE DB: {e}")
        logger.debug(traceback.format_exc())
        return

if __name__ == "__main__":
    main()