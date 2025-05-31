import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "c_rag_ex"

list_of_files = [
    "data/.gitkeep",
    "chromadb/.gitkeep",
    
    "RAG_Agent.ipynb",
    "main.py",
    "test_rag.py",
    
    "utils/__init__.py",
    "utils/config.py",
    "utils/logger.py",
    "utils/common.py",
    
    "params.yaml",
    ".env.local",
    ".env.example",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")