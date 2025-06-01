from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_ollama import OllamaLLM
from utils.config import CONFIG

LOCAL_LLM = CONFIG["LOCAL_LLM"]

def embedding_func():
    return GPT4AllEmbeddings()

def llm_func(prompt):
    llm = OllamaLLM(
        model=LOCAL_LLM,
        format="json",
        temperature=0.1,
        max_tokens=512,
        # streaming=True,
        streaming=False,
        verbose=True,
    )
    return llm.invoke(prompt)