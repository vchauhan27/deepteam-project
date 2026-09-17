# Research Agent

                                           USER
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ Research Agent  │
                                   │   LangChain     │
                                   └────────┬────────┘
                                            │
                                  decides which tool
                                            │
                  ┌──────────────────┬──────┴───────┬──────────────────┐
                  │                  │              │                  │
                  ▼                  ▼              ▼                  ▼
          ┌───────────────┐  ┌──────────────┐ ┌──────────────┐  ┌──────────────┐
          │ RAG Tool      │  │ Web Tool     │ │ Memory Tool  │  │ MCP Tools    │
          │               │  │              │ │              │  │              │
          │ BGE-M3        │  │ Tavily       │ │ LangGraph    │  │ FastMCP      │
          │      ↓        │  │      ↓       │ │    Store     │  │   Server     │
          │    Chroma     │  │    Web       │ │              │  │              │
          └───────┬───────┘  └──────┬───────┘ └──────┬───────┘  └──────┬───────┘
                  │                 │                │                 │
                  └─────────────────┴────────┬───────┴─────────────────┘
                                             ▼
                                    ┌─────────────────┐
                                    │    OpenRouter   │
                                    │   Chat Model    │
                                    └────────┬────────┘
                                             │
                                             ▼
                                           ANSWER

This directory contains the `research-agent`, an intelligent Agentic RAG implementation built using LangChain and LangGraph. It is designed to evaluate both internal knowledge base documents and external web sources to formulate well-reasoned answers to user queries.

## Architecture & Tools

The agent is driven by a chat model via OpenRouter and acts as a central decision-maker. It is equipped with several tools:

- **RAG Tool (`retrieve_documents`)**: Queries an internal knowledge base stored in a Chroma vector database using BGE-M3 embeddings.
- **Web Tool (`search_web`)**: Queries the public internet using Tavily Search when the question requires current or external information.
- **Memory Tools (`remember_fact` & `recall_facts`)**: Utilizes LangGraph's `InMemoryStore` to store and retrieve long-term facts about the user across conversations.

## MCP Server Integration

The research agent also includes a FastMCP server (`mcp_server.py`) which exposes utility tools for external clients over the `stdio` transport. 
Current tools provided by the MCP server include:
- `word_count`: Count words, characters, and estimate reading time.
- `format_citation`: Format source citations into APA or MLA styles.

## DeepEval Safety Testing

The agent integrates with the DeepEval framework to perform automated adversarial safety tests to ensure it stays in-bounds and adheres to its role.
Evaluations are located in `../llm-eval/safety-eval/` and test the agent for:
- **Bias & Toxicity**
- **Misuse**: Ensuring it doesn't deviate from research (e.g. refusing to plan vacations or write poetry).
- **PII Leakage**: Refusing to summarize sensitive personal data.
- **Non-Advice**: Refusing to give medical, legal, or financial advice.
- **Role Violation**: Maintaining the persona of an academic research agent.

## How It Works

1. The user asks a research question.
2. The agent interprets the query and decides which tools to invoke based on its system prompt.
3. It may retrieve internal documents, perform web searches, or both.
4. It synthesizes the retrieved information to provide a comprehensive answer, maintaining context (via memory checkpoints) across multiple interactions.

## Usage

To run the agent interactively via the CLI:

```bash
python agent.py
```
You will be prompted to enter research questions in a loop. Type `exit` or `quit` to end the session.

To run the MCP server:
```bash
python mcp_server.py
```

## Configuration

Make sure your `.env` file is properly configured with the necessary API keys:
- `OPENROUTER_API_KEY`
- `TAVILY_API_KEY`
- `CONFIDENTAI_API_KEY` (for DeepEval)

Model selection for the agent and embeddings is managed centrally in the `config.py` file located at the project root.