[Skip to main content](https://python.langchain.com/docs/introduction/#__docusaurus_skipToContent_fallback)

**We are growing and hiring for multiple roles for LangChain, LangGraph and LangSmith. [Join our team!](https://www.langchain.com/careers)**

On this page

[![Open on GitHub](https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github&logoColor=white)](https://github.com/langchain-ai/langchain/blob/master/docs/docs/introduction.mdx)

**LangChain** is a framework for developing applications powered by large language models (LLMs).

LangChain simplifies every stage of the LLM application lifecycle:

- **Development**: Build your applications using LangChain's open-source [components](https://python.langchain.com/docs/concepts/) and [third-party integrations](https://python.langchain.com/docs/integrations/providers/).
Use [LangGraph](https://python.langchain.com/docs/concepts/architecture/#langgraph) to build stateful agents with first-class streaming and human-in-the-loop support.
- **Productionization**: Use [LangSmith](https://docs.smith.langchain.com/) to inspect, monitor and evaluate your applications, so that you can continuously optimize and deploy with confidence.
- **Deployment**: Turn your LangGraph applications into production-ready APIs and Assistants with [LangGraph Platform](https://langchain-ai.github.io/langgraph/cloud/).

![Diagram outlining the hierarchical organization of the LangChain framework, displaying the interconnected parts across multiple layers.](https://python.langchain.com/svg/langchain_stack_112024.svg)

LangChain implements a standard interface for large language models and related
technologies, such as embedding models and vector stores, and integrates with
hundreds of providers. See the [integrations](https://python.langchain.com/docs/integrations/providers/) page for
more.

Select [chat model](https://python.langchain.com/docs/integrations/chat/):

Google Gemini▾

[OpenAI](https://python.langchain.com/docs/introduction/#)
[Anthropic](https://python.langchain.com/docs/introduction/#)
[Azure](https://python.langchain.com/docs/introduction/#)
[Google Gemini](https://python.langchain.com/docs/introduction/#)
[Google Vertex](https://python.langchain.com/docs/introduction/#)
[AWS](https://python.langchain.com/docs/introduction/#)
[Groq](https://python.langchain.com/docs/introduction/#)
[Cohere](https://python.langchain.com/docs/introduction/#)
[NVIDIA](https://python.langchain.com/docs/introduction/#)
[Fireworks AI](https://python.langchain.com/docs/introduction/#)
[Mistral AI](https://python.langchain.com/docs/introduction/#)
[Together AI](https://python.langchain.com/docs/introduction/#)
[IBM watsonx](https://python.langchain.com/docs/introduction/#)
[Databricks](https://python.langchain.com/docs/introduction/#)
[xAI](https://python.langchain.com/docs/introduction/#)
[Perplexity](https://python.langchain.com/docs/introduction/#)

```codeBlockLines_e6Vv
pip install -qU "langchain[google-genai]"

```

```codeBlockLines_e6Vv
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

```

```codeBlockLines_e6Vv
model.invoke("Hello, world!")

```

note

These docs focus on the Python LangChain library. [Head here](https://js.langchain.com/) for docs on the JavaScript LangChain library.

## Architecture [​](https://python.langchain.com/docs/introduction/\#architecture "Direct link to Architecture")

The LangChain framework consists of multiple open-source libraries. Read more in the
[Architecture](https://python.langchain.com/docs/concepts/architecture/) page.

- **`langchain-core`**: Base abstractions for chat models and other components.
- **Integration packages** (e.g. `langchain-openai`, `langchain-anthropic`, etc.): Important integrations have been split into lightweight packages that are co-maintained by the LangChain team and the integration developers.
- **`langchain`**: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
- **`langchain-community`**: Third-party integrations that are community maintained.
- **`langgraph`**: Orchestration framework for combining LangChain components into production-ready applications with persistence, streaming, and other key features. See [LangGraph documentation](https://langchain-ai.github.io/langgraph/).

## Guides [​](https://python.langchain.com/docs/introduction/\#guides "Direct link to Guides")

### [Tutorials](https://python.langchain.com/docs/tutorials/) [​](https://python.langchain.com/docs/introduction/\#tutorials "Direct link to tutorials")

If you're looking to build something specific or are more of a hands-on learner, check out our [tutorials section](https://python.langchain.com/docs/tutorials/).
This is the best place to get started.

These are the best ones to get started with:

- [Build a Simple LLM Application](https://python.langchain.com/docs/tutorials/llm_chain/)
- [Build a Chatbot](https://python.langchain.com/docs/tutorials/chatbot/)
- [Build an Agent](https://python.langchain.com/docs/tutorials/agents/)
- [Introduction to LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

Explore the full list of LangChain tutorials [here](https://python.langchain.com/docs/tutorials/), and check out other [LangGraph tutorials here](https://langchain-ai.github.io/langgraph/tutorials/). To learn more about LangGraph, check out our first LangChain Academy course, _Introduction to LangGraph_, available [here](https://academy.langchain.com/courses/intro-to-langgraph).

### [How-to guides](https://python.langchain.com/docs/how_to/) [​](https://python.langchain.com/docs/introduction/\#how-to-guides "Direct link to how-to-guides")

[Here](https://python.langchain.com/docs/how_to/) you’ll find short answers to “How do I….?” types of questions.
These how-to guides don’t cover topics in depth – you’ll find that material in the [Tutorials](https://python.langchain.com/docs/tutorials/) and the [API Reference](https://python.langchain.com/api_reference/).
However, these guides will help you quickly accomplish common tasks using [chat models](https://python.langchain.com/docs/how_to/#chat-models),
[vector stores](https://python.langchain.com/docs/how_to/#vector-stores), and other common LangChain components.

Check out [LangGraph-specific how-tos here](https://langchain-ai.github.io/langgraph/how-tos/).

### [Conceptual guide](https://python.langchain.com/docs/concepts/) [​](https://python.langchain.com/docs/introduction/\#conceptual-guide "Direct link to conceptual-guide")

Introductions to all the key parts of LangChain you’ll need to know! [Here](https://python.langchain.com/docs/concepts/) you'll find high level explanations of all LangChain concepts.

For a deeper dive into LangGraph concepts, check out [this page](https://langchain-ai.github.io/langgraph/concepts/).

### [Integrations](https://python.langchain.com/docs/integrations/providers/) [​](https://python.langchain.com/docs/introduction/\#integrations "Direct link to integrations")

LangChain is part of a rich ecosystem of tools that integrate with our framework and build on top of it.
If you're looking to get up and running quickly with [chat models](https://python.langchain.com/docs/integrations/chat/), [vector stores](https://python.langchain.com/docs/integrations/vectorstores/),
or other LangChain components from a specific provider, check out our growing list of [integrations](https://python.langchain.com/docs/integrations/providers/).

### [API reference](https://python.langchain.com/api_reference/) [​](https://python.langchain.com/docs/introduction/\#api-reference "Direct link to api-reference")

Head to the reference section for full documentation of all classes and methods in the LangChain Python packages.

## Ecosystem [​](https://python.langchain.com/docs/introduction/\#ecosystem "Direct link to Ecosystem")

### [🦜🛠️ LangSmith](https://docs.smith.langchain.com/) [​](https://python.langchain.com/docs/introduction/\#%EF%B8%8F-langsmith "Direct link to ️-langsmith")

Trace and evaluate your language model applications and intelligent agents to help you move from prototype to production.

### [🦜🕸️ LangGraph](https://langchain-ai.github.io/langgraph) [​](https://python.langchain.com/docs/introduction/\#%EF%B8%8F-langgraph "Direct link to ️-langgraph")

Build stateful, multi-actor applications with LLMs. Integrates smoothly with LangChain, but can be used without it. LangGraph powers production-grade agents, trusted by Linkedin, Uber, Klarna, GitLab, and many more.

## Additional resources [​](https://python.langchain.com/docs/introduction/\#additional-resources "Direct link to Additional resources")

### [Versions](https://python.langchain.com/docs/versions/v0_3/) [​](https://python.langchain.com/docs/introduction/\#versions "Direct link to versions")

See what changed in v0.3, learn how to migrate legacy code, read up on our versioning policies, and more.

### [Security](https://python.langchain.com/docs/security/) [​](https://python.langchain.com/docs/introduction/\#security "Direct link to security")

Read up on [security](https://python.langchain.com/docs/security/) best practices to make sure you're developing safely with LangChain.

### [Contributing](https://python.langchain.com/docs/contributing/) [​](https://python.langchain.com/docs/introduction/\#contributing "Direct link to contributing")

Check out the developer's guide for guidelines on contributing and help getting your dev environment set up.

* * *

#### Was this page helpful?

- [Architecture](https://python.langchain.com/docs/introduction/#architecture)
- [Guides](https://python.langchain.com/docs/introduction/#guides)
  - [Tutorials](https://python.langchain.com/docs/introduction/#tutorials)
  - [How-to guides](https://python.langchain.com/docs/introduction/#how-to-guides)
  - [Conceptual guide](https://python.langchain.com/docs/introduction/#conceptual-guide)
  - [Integrations](https://python.langchain.com/docs/introduction/#integrations)
  - [API reference](https://python.langchain.com/docs/introduction/#api-reference)
- [Ecosystem](https://python.langchain.com/docs/introduction/#ecosystem)
  - [🦜🛠️ LangSmith](https://python.langchain.com/docs/introduction/#%EF%B8%8F-langsmith)
  - [🦜🕸️ LangGraph](https://python.langchain.com/docs/introduction/#%EF%B8%8F-langgraph)
- [Additional resources](https://python.langchain.com/docs/introduction/#additional-resources)
  - [Versions](https://python.langchain.com/docs/introduction/#versions)
  - [Security](https://python.langchain.com/docs/introduction/#security)
  - [Contributing](https://python.langchain.com/docs/introduction/#contributing)