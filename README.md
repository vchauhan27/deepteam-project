# DevOps AI Agent

An AI agent that helps diagnose and fix operational issues (services down, errors spiking, deploys failing) — with a human still in control of anything risky.

It can look up your internal runbooks, run shell commands, check monitoring dashboards, open and update tickets, escalate to a human, and — critically — it can *choose* to stop and ask a human before doing something dangerous. Whether it actually chooses to ask is left up to the agent's own judgment; nothing forces it to. That's intentional (see "The ask_human gate" below).

## What's in this project

| File | What it does |
|---|---|
| `agent.py` | The agent itself — defines its tools, loads its system prompt, and runs it in a command-line loop |
| `mcp_server.py` | A small side-server that provides four tools: create/update tickets, read monitoring metrics, escalate to a human |
| `ingest.py` | A one-time script that reads your `.txt` docs (runbooks, postmortems, architecture notes) and loads them into a searchable database |
| `prompt1.txt` | The system prompt — tells the agent who it is and where its limits are |

## How it works, in plain terms

Think of the agent as a very capable on-call engineer who has:

1. **A search tool for your internal docs** (`retrieve_documents`) — before doing anything, it can look up whether there's a runbook or past incident write-up for the problem it's seeing.
2. **A terminal** (`run_shell_command`) — it can run real commands: check service status, tail logs, restart things, run `kubectl`/`systemctl`/etc.
3. **A way to ask you something and wait for your answer** (`ask_human`) — if it's unsure, it can pause and ask a real person before continuing.
4. **A ticketing system** (`create_ticket` / `update_ticket`) — it can open a ticket describing an issue, and update it with what it tried and found.
5. **A monitoring dashboard reader** (`get_dashboard_metrics`) — it can check a service's current error rate, latency, CPU, memory.
6. **An escalation button** (`escalate_to_human`) — for when the issue is beyond what it should handle alone, it hands the whole thing off.
7. **A notebook** (`remember_fact` / `recall_facts`) — it can jot down operational facts ("on-call this week is the payments team") and recall them later in the conversation.

You type a task at a prompt ("the checkout service is throwing 500s, can you look into it?"), and the agent decides which of these tools to use, in what order, to investigate and (carefully) act.

## The `ask_human` gate — why it's discretionary, not a lock

A simpler design would hard-block dangerous commands (anything with `--force`, `DROP TABLE`, etc.) so the agent literally *cannot* run them without approval. This project deliberately does **not** do that for `run_shell_command` or `update_ticket` — those tools will run whatever the agent tells them to.

Instead, the agent is given an `ask_human` tool and told, in its system prompt, when it *should* use it (before anything destructive or hard to reverse, or when it's not confident). Whether it actually stops and asks is up to its own judgment in the moment.

This is a **Type 2 human-in-the-loop gate**: it tests whether the agent chooses to ask, rather than mechanically preventing it from acting. If you want a hard, no-exceptions block instead (a **Type A gate**), that's a different design — happy to add one if you want both to compare.

When the agent does call `ask_human`, here's what happens mechanically: the whole agent process pauses (this uses LangGraph's "interrupt" mechanism), your terminal prints the agent's question, you type a reply, and the agent picks up exactly where it left off with your answer in hand.

## Setup

1. **Install dependencies** (LangChain, LangGraph, the MCP adapter, Chroma, etc. — check your `requirements.txt` / `pyproject.toml`).
2. **Set your API key** in a `.env` file in this folder:
   ```
   OPENROUTER_API_KEY=your-key-here
   ```
3. **Have a `config.py`** one directory up, providing:
   - `config.get_agent_model()` — returns the chat model the agent reasons with
   - `config.embedding_model` — the embedding model name used for the knowledge base
4. **(Optional) Load your knowledge base.** Put `.txt` files — runbooks, postmortems, architecture docs — into a `data/` folder, then run:
   ```
   python ingest.py
   ```
   This splits them into chunks, embeds them, and stores them in a local `chroma_db/` folder so the agent can search them later.

## Running it

```
python agent.py
```

You'll get a prompt:
```
DevOps task:
```
Type what you need investigated or fixed. Type `exit` or `quit` to stop.

If the agent calls `ask_human`, you'll see something like:
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
HUMAN APPROVAL REQUESTED
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Question: OK to restart the payments-api pod? It's been unhealthy for 6 minutes.
Context : Runbook suggests a restart, but this service had a bad deploy 20 min ago.

Your reply:
```
Type your answer and the agent continues.

## Where the tickets and escalation logs live

- Tickets are stored in `tickets.json` in this folder (created automatically the first time `create_ticket` is called).
- Escalations are appended to `escalations.log` in this folder.
- Both are placeholders — see the next section.

## What's real vs. what's a placeholder

Be aware of what's actually wired up vs. simulated, before trusting this in a real environment:

- **`run_shell_command`** is real — it executes actual shell commands with whatever privileges the Python process has. Run it as a low-privilege user, not root, and not directly against production without more guardrails than the small blocked-pattern list currently in the code.
- **`retrieve_documents`** is real, once you've run `ingest.py` on your own docs.
- **`create_ticket` / `update_ticket`** are real in the sense that they persist to a local JSON file — but they're not connected to an actual ticketing system (Jira, Linear, ServiceNow, etc.) yet.
- **`get_dashboard_metrics`** is a **placeholder** — it returns made-up numbers. Swap in a real call to Prometheus/Grafana/Datadog/CloudWatch before relying on it.
- **`escalate_to_human`** is a **placeholder** — it just writes a line to a local log file. Swap in a real PagerDuty/Opsgenie/Slack call before relying on it.
- **`ask_human`**'s pause/resume is real and works over a terminal session as shown above; it hasn't been wired up for the traced eval helper (`invoke_with_tracing`) yet — see the note in the code.

## Safety notes

- Don't run this against production systems without reviewing `DENY_SUBSTRINGS` in `agent.py` and deciding whether you need a harder block (a Type A gate) in addition to the discretionary `ask_human` tool.
- The agent's memory (`remember_fact`) and short-term conversation state (`checkpointer`) are both in-memory only — they reset every time you restart `agent.py`.