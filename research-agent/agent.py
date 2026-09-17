import os
import sys
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_mcp_adapters.client import MultiServerMCPClient

from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY is not set.")

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

# ---------------------------------------------------------
# 1. OpenRouter reasoning model
# ---------------------------------------------------------

model = config.get_agent_model()

# ---------------------------------------------------------
# 2. BGE-M3 embeddings
# ---------------------------------------------------------


class OpenRouterEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts, chunk_size=None, **kwargs):
        embeddings = []
        for text in texts:
            response = self.client.create(model=self.model, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text, **kwargs):
        response = self.client.create(model=self.model, input=text)
        return response.data[0].embedding


embeddings = OpenRouterEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,  # type: ignore
    model=config.embedding_model,
)

# ---------------------------------------------------------
# 3. Chroma vector database
# ---------------------------------------------------------

vectorstore = Chroma(
    collection_name="research_docs",
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)

# short memory
checkpointer = InMemorySaver()

# ---------------------------------------------------------
# 4. Retriever
# ---------------------------------------------------------

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ---------------------------------------------------------
# 5. RAG tool
# ---------------------------------------------------------


@tool
def retrieve_documents(query: str) -> str:
    """
    Search the internal research knowledge base.

    Use this tool when the question may be answered using
    information contained in the internal documents.
    """

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant internal documents were found."

    results = []

    for i, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")

        results.append(
            f"""
SOURCE TYPE: INTERNAL KNOWLEDGE BASE
DOCUMENT: {i}
SOURCE: {source}

CONTENT:
{document.page_content}
"""
        )

    return "\n".join(results)


# ---------------------------------------------------------
# 6. Web search
# ---------------------------------------------------------

web_search = TavilySearch(max_results=5)


@tool
def search_web(query: str) -> str:
    """
    Search the public web.

    Use this tool for current, recent, external, or
    time-sensitive information that may not exist in
    the internal knowledge base.
    """

    result = web_search.invoke({"query": query})

    return str(result)

# ---------------------------------------------------------
# 7. MCP utility tools (word_count, format_citation) — served
#    by mcp_server.py, nothing to do with retrieval.
# ---------------------------------------------------------

mcp_client = MultiServerMCPClient(
    {
        "utils": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(BASE_DIR / "mcp_server.py")],
        }
    }
)

mcp_tools = asyncio.run(mcp_client.get_tools())

# long memory
store = InMemoryStore()


@dataclass
class Context:
    user_id: str


@tool
def remember_fact(fact: str, runtime: ToolRuntime[Context]) -> str:
    """
    Save a fact about the user or their preferences for future
    conversations (e.g. "prefers concise answers", "researching RAG systems").
    """
    if runtime.store is None:
        return "Memory store is not available."
    key = str(hash(fact))[:8]
    runtime.store.put(("user_facts", runtime.context.user_id), key, {"fact": fact})
    return "Saved."


@tool
def recall_facts(runtime: ToolRuntime[Context]) -> str:
    """Retrieve previously saved facts about the user."""
    if runtime.store is None:
        return "Memory store is not available."
    items = runtime.store.search(("user_facts", runtime.context.user_id))
    if not items:
        return "No saved facts about this user yet."
    return "\n".join(item.value["fact"] for item in items)


# ---------------------------------------------------------
# 8. Agent system prompt
# ---------------------------------------------------------

#   prompt1.txt -- original prompt (no explicit planning step)
#   prompt2.txt -- current prompt (adds a "state your plan" instruction)

PROMPT_PATH = BASE_DIR / "prompt1.txt"

SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------
# 9. Create agent
# ---------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[
        retrieve_documents,
        search_web,
        *mcp_tools,
        remember_fact,
        recall_facts,
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    store=store,
    context_schema=Context,
)


# ---------------------------------------------------------
# 9b. Traced invocation helper (for DeepEval agentic metrics)
# ---------------------------------------------------------

def invoke_with_tracing(question: str, thread_id: str = "default", user_id: str = "default", agent_instance=None):
    from deepeval.integrations.langchain import CallbackHandler

    target_agent = agent_instance or agent

    return target_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [CallbackHandler()],
        },
        context=Context(user_id=user_id),
    )


# ---------------------------------------------------------
# 10. Run agent
# ---------------------------------------------------------


def main():

    print("=" * 70)
    print("RESEARCH AGENT")
    print("=" * 70)

    print("\nAvailable tools:")
    print("  - retrieve_documents")
    print("  - search_web")
    for t in mcp_tools:
        print(f"  - {t.name} (MCP)")
    print("  - remember_fact")
    print("  - recall_facts")

    while True:
        print("\n" + "-" * 70)

        question = input("Research question: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("\nExiting.")
            break

        print("\nAgent is researching...\n")

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": "cli-session"}},
                context=Context(user_id="cli-user"),
            )

            print("=" * 70)
            print("ANSWER")
            print("=" * 70)
            print(result["messages"][-1].content)

        except Exception as e:
            print("\nAgent error:")
            print(repr(e))


if __name__ == "__main__":
    main()