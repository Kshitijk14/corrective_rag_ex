[Skip to content](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag_local.ipynb#start-of-content)

[langchain-ai](https://github.com/langchain-ai)/ **[langgraph](https://github.com/langchain-ai/langgraph)** Public

- [Notifications](https://github.com/login?return_to=%2Flangchain-ai%2Flanggraph) You must be signed in to change notification settings
- [Fork\\
2.3k](https://github.com/login?return_to=%2Flangchain-ai%2Flanggraph)
- [Star\\
13.4k](https://github.com/login?return_to=%2Flangchain-ai%2Flanggraph)


## Files

main

/

# langgraph\_self\_rag\_local.ipynb

Copy path

Blame

Blame

## Latest commit

[![jujumilk3](https://avatars.githubusercontent.com/u/41659814?v=4&size=40)](https://github.com/jujumilk3)[jujumilk3](https://github.com/langchain-ai/langgraph/commits?author=jujumilk3)

[docs(rag): fix many typo of generate at RAG section (](https://github.com/langchain-ai/langgraph/commit/a6e2d9e1976ecd907ebf03d62bb036796c710546) [#4490](https://github.com/langchain-ai/langgraph/pull/4490) [)](https://github.com/langchain-ai/langgraph/commit/a6e2d9e1976ecd907ebf03d62bb036796c710546)

May 1, 2025

[a6e2d9e](https://github.com/langchain-ai/langgraph/commit/a6e2d9e1976ecd907ebf03d62bb036796c710546) · May 1, 2025

## History

[History](https://github.com/langchain-ai/langgraph/commits/main/examples/rag/langgraph_self_rag_local.ipynb)

747 lines (747 loc) · 242 KB

/

# langgraph\_self\_rag\_local.ipynb

Top

## File metadata and controls

- Preview

- Code

- Blame


747 lines (747 loc) · 242 KB

[Raw](https://github.com/langchain-ai/langgraph/raw/refs/heads/main/examples/rag/langgraph_self_rag_local.ipynb)

Loading

Notebooks

# Self-RAG using local LLMs [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Self-RAG-using-local-LLMs)

Self-RAG is a strategy for RAG that incorporates self-reflection / self-grading on retrieved documents and generations.

In the [paper](https://arxiv.org/abs/2310.11511), a few decisions are made:

1. Should I retrieve from retriever, `R` -

- Input: `x (question)` OR `x (question)`, `y (generation)`
- Decides when to retrieve `D` chunks with `R`
- Output: `yes, no, continue`

2. Are the retrieved passages `D` relevant to the question `x` -

- Input: ( `x (question)`, `d (chunk)`) for `d` in `D`
- `d` provides useful information to solve `x`
- Output: `relevant, irrelevant`

3. Are the LLM generation from each chunk in `D` is relevant to the chunk (hallucinations, etc) -

- Input: `x (question)`, `d (chunk)`, `y (generation)` for `d` in `D`
- All of the verification-worthy statements in `y (generation)` are supported by `d`
- Output: `{fully supported, partially supported, no support`

4. The LLM generation from each chunk in `D` is a useful response to `x (question)` -

- Input: `x (question)`, `y (generation)` for `d` in `D`
- `y (generation)` is a useful response to `x (question)`.
- Output: `{5, 4, 3, 2, 1}`

We will implement some of these ideas from scratch using [LangGraph](https://langchain-ai.github.io/langgraph/).

![Screenshot 2024-04-01 at 12.42.59 PM.png](<Base64-Image-Removed>)

## Setup [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Setup)

First let's install our required packages and set our API keys

In \[ \]:

```
%capture --no-stderr
%pip install -U langchain-nomic langchain_community tiktoken langchainhub chromadb langchain langgraph nomic[local]

```

In \[ \]:

```
import getpass
import os

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")

_set_env("NOMIC_API_KEY")

```

Set up [LangSmith](https://smith.langchain.com/) for LangGraph development

Sign up for LangSmith to quickly spot issues and improve the performance of your LangGraph projects. LangSmith lets you use trace data to debug, test, and monitor your LLM apps built with LangGraph — read more about how to get started [here](https://docs.smith.langchain.com/).


### LLMs [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#LLMs)

#### Local Embeddings [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Local-Embeddings)

You can use `GPT4AllEmbeddings()` from Nomic, which can access use Nomic's recently released [v1](https://blog.nomic.ai/posts/nomic-embed-text-v1) and [v1.5](https://blog.nomic.ai/posts/nomic-embed-matryoshka) embeddings.

Follow the documentation [here](https://docs.gpt4all.io/gpt4all_python_embedding.html#supported-embedding-models).

#### Local LLM [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Local-LLM)

(1) Download [Ollama app](https://ollama.ai/).

(2) Download a `Mistral` model from various Mistral versions [here](https://ollama.ai/library/mistral) and Mixtral versions [here](https://ollama.ai/library/mixtral) available.

```
ollama pull mistral

```

In \[2\]:

```
# Ollama model name
local_llm = "mistral"

```

## Create Index [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Create-Index)

Let's index 3 blog posts.

In \[ \]:

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_nomic.embeddings import NomicEmbeddings

urls = [\
    "https://lilianweng.github.io/posts/2023-06-23-agent/",\
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",\
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",\
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)
retriever = vectorstore.as_retriever()

```

## LLMs [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#LLMs)

In \[6\]:

```
### Retrieval Grader

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()
question = "agent memory"
docs = retriever.get_relevant_documents(question)
doc_txt = docs[1].page_content
print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

```

```
{'score': 'yes'}

```

In \[7\]:

```
### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOllama(model=local_llm, temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run
generation = rag_chain.invoke({"context": docs, "question": question})
print(generation)

```

```
 In an LLM-powered autonomous agent system, the Large Language Model (LLM) functions as the agent's brain. The agent has key components including memory, planning, and reflection mechanisms. The memory component is a long-term memory module that records a comprehensive list of agents’ experience in natural language. It includes a memory stream, which is an external database for storing past experiences. The reflection mechanism synthesizes memories into higher-level inferences over time and guides the agent's future behavior.

```

In \[8\]:

```
### Hallucination Grader

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()
hallucination_grader.invoke({"documents": docs, "generation": generation})

```

Out\[8\]:

```
{'score': 'yes'}
```

In \[10\]:

```
### Answer Grader

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n
    Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()
answer_grader.invoke({"question": question, "generation": generation})

```

Out\[10\]:

```
{'score': 'yes'}
```

In \[11\]:

```
### Question Re-writer

# LLM
llm = ChatOllama(model=local_llm, temperature=0)

# Prompt
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})

```

Out\[11\]:

```
' What is agent memory and how can it be effectively utilized in vector database retrieval?'
```

# Graph [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Graph)

Capture the flow in as a graph.

## Graph state [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Graph-state)

In \[13\]:

```
from typing import List

from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

```

In \[14\]:

```
### Nodes

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

### Edges

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

```

## Build Graph [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Build-Graph)

This just follows the flow we outlined in the figure above.

In \[15\]:

```
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

```

## Run [¶](https://github.com/langchain-ai/langgraph/blob/765bc3b9e05f36bd6775f6ad3121d6ac650c9a61/examples/rag/\#Run)

In \[16\]:

```
from pprint import pprint

# Run
inputs = {"question": "Explain how the different types of agent memory work?"}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint(f"Node '{key}':")
        # Optional: print full state at each node
        # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
    pprint("\n---\n")

# Final generation
pprint(value["generation"])

```

```
---RETRIEVE---
"Node 'retrieve':"
'\n---\n'
---CHECK DOCUMENT RELEVANCE TO QUESTION---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
---GRADE: DOCUMENT RELEVANT---
"Node 'grade_documents':"
'\n---\n'
---ASSESS GRADED DOCUMENTS---
---DECISION: GENERATE---
---GENERATE---
"Node 'generate':"
'\n---\n'
---CHECK HALLUCINATIONS---
---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---
---GRADE GENERATION vs QUESTION---
---DECISION: GENERATION ADDRESSES QUESTION---
"Node '__end__':"
'\n---\n'
(' In a LLM-powered autonomous agent system, memory is a key component that '
 'enables agents to store and retrieve information. There are different types '
 'of memory in human brains, such as sensory memory which retains impressions '
 'of sensory information for a few seconds, and long-term memory which records '
 "experiences for extended periods (Lil'Log, 2023). In the context of LLM "
 'agents, memory is often implemented as an external database or memory stream '
 "(Lil'Log, 2023). The agent can consult this memory to inform its behavior "
 'based on relevance, recency, and importance. Additionally, reflection '
 'mechanisms synthesize memories into higher-level inferences over time and '
 "guide the agent's future behavior (Lil'Log, 2023).")

```

Trace:

[https://smith.langchain.com/public/4163a342-5260-4852-8602-bda3f95177e7/r](https://smith.langchain.com/public/4163a342-5260-4852-8602-bda3f95177e7/r)