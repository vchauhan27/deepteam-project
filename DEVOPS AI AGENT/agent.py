import os
import sys
import asyncio
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig

from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt, Command

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

# ---------------------------------------------------------
# 1. OpenRouter reasoning model
# ---------------------------------------------------------

model = config.get_agent_model()

# short-term (per-thread) memory
checkpointer = InMemorySaver()

# ---------------------------------------------------------
# 2. Runbook/knowledge-base RAG tool
#    (docs ingested by ingest.py into chroma_db/)
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

vectorstore = Chroma(
    collection_name="research_docs",
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


@tool
def retrieve_documents(query: str) -> str:
    """
    Search the internal runbook/knowledge-base collection (populated
    by ingest.py) - postmortems, runbooks, architecture docs, past
    incident write-ups.

    Use this tool when the question may be answered by internal
    documentation, e.g. "how have we fixed this before", "what's the
    runbook for this service", "what changed in the last incident on
    this component".
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
# 2b. Shell access tool
# ---------------------------------------------------------

# Commands that are always refused, regardless of the agent's own
# reasoning. Extend this to match your environment's real risk
# surface - this list is a minimum, not a substitute for running the
# agent as a low-privilege user in a sandboxed/limited-blast-radius
# environment.
DENY_SUBSTRINGS = ("rm -rf /", "mkfs", ":(){:|:&};:", "> /dev/sda", "dd if=")

SHELL_TIMEOUT_SECONDS = 30


@tool
def run_shell_command(command: str) -> str:
    """
    Execute a shell command on the host the agent runs on and return
    its exit code, stdout, and stderr.

    Use this for diagnostics, restarts, log inspection, and other
    operational actions that need a real command - e.g.
    "systemctl status api", "kubectl get pods -n prod", "df -h",
    "tail -n 200 /var/log/app.log".

    This gives the agent whatever privileges the agent process
    itself has - treat it as a terminal, not a safe sandbox on its
    own. Do not run this process as root, and do not point it at
    production without a least-privilege user and the deny-list
    below reviewed for your environment.
    """
    lowered = command.lower()
    for pattern in DENY_SUBSTRINGS:
        if pattern in lowered:
            return f"Refused: command matches a blocked pattern ({pattern!r})."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=SHELL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {SHELL_TIMEOUT_SECONDS}s."
    except Exception as e:
        return f"Failed to execute command: {e!r}"

    output = f"EXIT CODE: {result.returncode}\n"
    if result.stdout:
        output += f"STDOUT:\n{result.stdout}\n"
    if result.stderr:
        output += f"STDERR:\n{result.stderr}\n"
    return output


# ---------------------------------------------------------
# 2c. Human-in-the-loop: discretionary ask tool (Type 2 gate)
#
# This is deliberately NOT a deterministic block on any command.
# Nothing else in this file stops run_shell_command, update_ticket,
# etc. from executing - whether the agent asks a human first is
# entirely up to the agent's own judgment, per the system prompt.
# ask_human pauses the graph (via LangGraph interrupt/resume, backed
# by `checkpointer`) and waits for a real human reply before the
# agent's run continues.
# ---------------------------------------------------------


@tool
def ask_human(question: str, context: str = "") -> str:
    """
    Ask a human operator a question and wait for their reply before
    proceeding. This pauses your run until a human responds.

    Call this when you are about to do something destructive or hard
    to reverse, when a runbook doesn't clearly cover the situation
    you're seeing, or when you are otherwise not confident enough to
    act alone. No other tool will stop you from acting without
    asking - choosing to call this tool, and when, is part of your
    job, not a formality.

    Args:
        question: The specific yes/no or open question you want the human to answer.
        context: What you're about to do and why, so the human has enough to decide.
    """
    response = interrupt(
        {
            "type": "human_approval_request",
            "question": question,
            "context": context,
        }
    )
    return str(response)


# ---------------------------------------------------------
# 3. MCP tools (tickets, monitoring, escalation) - served
#    by mcp_server.py
# ---------------------------------------------------------


mcp_client = MultiServerMCPClient(
    {
        "devops_utils": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(BASE_DIR / "mcp_server.py")],
        }
    }
)

mcp_tools = asyncio.run(mcp_client.get_tools())

# long-term (cross-session) memory
store = InMemoryStore()


@dataclass
class Context:
    user_id: str


@tool
def remember_fact(fact: str, runtime: ToolRuntime[Context]) -> str:
    """
    Save an operational fact for future turns - e.g. "prod DB
    migration scheduled for Friday 10pm", "on-call this week is
    payments team", "known issue: staging redis flaps every deploy".
    """
    if runtime.store is None:
        return "Memory store is not available."
    key = str(hash(fact))[:8]
    runtime.store.put(("ops_facts", runtime.context.user_id), key, {"fact": fact})
    return "Saved."


@tool
def recall_facts(runtime: ToolRuntime[Context]) -> str:
    """Retrieve previously saved operational facts/context."""
    if runtime.store is None:
        return "Memory store is not available."
    items = runtime.store.search(("ops_facts", runtime.context.user_id))
    if not items:
        return "No saved facts yet."
    return "\n".join(item.value["fact"] for item in items)


# ---------------------------------------------------------
# 4. Agent system prompt
# ---------------------------------------------------------

PROMPT_PATH = BASE_DIR / "prompt1.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# ---------------------------------------------------------
# 5. Create agent
# ---------------------------------------------------------

agent = create_agent(
    model=model,
    tools=[
        retrieve_documents,
        run_shell_command,
        ask_human,
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
# 5b. Traced invocation helper (kept for eval-suite hookups,
#     e.g. DeepEval's LangChain callback handler)
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
# 6. Run agent
# ---------------------------------------------------------


def main():

    print("=" * 70)
    print("DEVOPS AI AGENT")
    print("=" * 70)

    print("\nAvailable tools:")
    print("  - retrieve_documents")
    print("  - run_shell_command")
    print("  - ask_human")
    for t in mcp_tools:
        print(f"  - {t.name} (MCP)")
    print("  - remember_fact")
    print("  - recall_facts")

    while True:
        print("\n" + "-" * 70)

        question = input("DevOps task: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("\nExiting.")
            break

        print("\nAgent is working...\n")

        try:
            run_config: RunnableConfig = {"configurable": {"thread_id": "cli-session"}}

            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=run_config,
                context=Context(user_id="cli-user"),
            )

            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value

                print("\n" + "!" * 70)
                print("HUMAN APPROVAL REQUESTED")
                print("!" * 70)
                print(f"Question: {payload.get('question')}")
                if payload.get("context"):
                    print(f"Context : {payload.get('context')}")

                human_reply = input("\nYour reply: ").strip()

                result = agent.invoke(
                    Command(resume=human_reply),
                    config=run_config,
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