import os
from pipeline.stage_01_populate_db import run_populate_db
from utils.config import CONFIG
from utils.logger import setup_logger
import argparse

URLS = CONFIG["URLS"]
LOG_PATH = CONFIG["LOG_PATH"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "main.log")

if __name__ == "__main__":
    logger = setup_logger("main_logger", LOG_FILE)
    
    logger.info(" ")
    logger.info("**********STARTING [STAGE 01]: POPULATE DB**********")
    # Create CLI.
    parser = argparse.ArgumentParser(description="Populate the database")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    run_populate_db(URLS, reset=args.reset)
    logger.info("**********FINISHED [STAGE 01]: POPULATE DB**********")
    logger.info(" ")
    
    