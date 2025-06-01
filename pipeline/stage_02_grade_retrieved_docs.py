import os
import traceback
import shutil
from dotenv import load_dotenv
from utils.config import CONFIG
from utils.logger import setup_logger
from langchain_chroma import Chroma
from utils.get_llm_func import embedding_func
from utils.get_llm_func import llm_func
from utils.get_prompt_temp import prompt_retrieval_grader
from langchain_core.output_parsers import JsonOutputParser


load_dotenv(dotenv_path=".env.local")

LOG_PATH = CONFIG["LOG_PATH"]
CHROMA_DB_PATH = CONFIG["CHROMA_DB_PATH"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_02_grade_retrieved_docs.log")

def query_rag(query_text: str, logger):
    try:
        logger.info("****[Stage 2] Querying Chroma DB****")
        # load the existing db (prep the db)
        db = Chroma(
            embedding_function=embedding_func(),
            persist_directory=CHROMA_DB_PATH,
        )
        logger.info(f"[Part 01] Loading existing DB from path: {CHROMA_DB_PATH}")
        
        # query the db (search the db)
        logger.info(f"[Part 02] Searching the db with text using similarity search: {query_text}")
        results = db.similarity_search_with_score(query_text, k=5)
        logger.info(f"[Result A] Found {len(results)} results: {results}")
        logger.info("****[Stage 2] Querying Chroma DB completed successfully****")
        
        logger.info("****[Stage 03] Grading Retrieved Docs****")
        retrieval_grader = prompt_retrieval_grader | llm_func | JsonOutputParser()
        graded_results = []

        for i, (doc, score) in enumerate(results):
            doc_text = doc.page_content
            logger.debug(f"[Part 03.{i}] Grading doc with score {score}")
            try:
                grade = retrieval_grader.invoke({
                    "question": query_text,
                    "document": doc_text,
                })
                graded_results.append((doc, score, grade["score"]))
                logger.info(f"[Part 03.{i}] Graded as: {grade['score']}")
            except Exception as grading_err:
                logger.error(f"[Part 03.{i}] Grading failed: {grading_err}")
                logger.debug(traceback.format_exc())

        logger.info(f"[Result B] Found {len(graded_results)} graded results: {graded_results}")
        logger.info("****[Stage 3] Grading completed successfully****")
        return graded_results
        
    except Exception as e:
        logger.error(f"Error in querying Chroma DB & grading: {e}")
        logger.debug(traceback.format_exc())
        return []

def run_grade_retrieved_docs():
    try:
        logger = setup_logger("retrieval_grader_logger", LOG_FILE)
        logger.info(" ")
        logger.info("++++++++Starting Retrieval Grading pipeline...")
        
        # Example query – replace with dynamic input if needed
        query = "how to save LLM costs?"
        results = query_rag(query, logger)

        logger.info("Graded results:")
        for idx, (doc, sim_score, relevance) in enumerate(results):
            logger.info(f"[Doc {idx}] Score: {sim_score:.4f}, Relevant: {relevance}")
            logger.debug(f"Content: {doc.page_content[:300]}...")
        
        logger.info("++++++++Retrieval Grading pipeline completed.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_grade_retrieved_docs()