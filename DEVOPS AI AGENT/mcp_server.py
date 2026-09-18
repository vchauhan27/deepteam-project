import json
import time
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("devops-agent-utils")

BASE_DIR = Path(__file__).resolve().parent
TICKETS_FILE = BASE_DIR / "tickets.json"
ESCALATIONS_LOG = BASE_DIR / "escalations.log"


# ---------------------------------------------------------
# Tiny local ticket store (JSON file). Swap this out for a
# real ticketing system (Jira, Linear, ServiceNow, ...) by
# replacing the load/save helpers and the two tools below.
# ---------------------------------------------------------


def _load_tickets() -> dict:
    if not TICKETS_FILE.exists():
        return {}
    return json.loads(TICKETS_FILE.read_text(encoding="utf-8"))


def _save_tickets(tickets: dict) -> None:
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2), encoding="utf-8")


@mcp.tool()
def create_ticket(title: str, description: str, severity: str = "medium") -> str:
    """
    Create a new incident/ops ticket.

    Use this tool when an issue needs to be tracked: a failing
    deploy, a recurring alert, a task to hand off to another team,
    or a plan you want a human to approve before you act on it.

    Args:
        title: Short summary of the issue.
        description: Full details - what happened, impact, context, and (if relevant) your proposed next step.
        severity: One of "low", "medium", "high", "critical".
    """
    tickets = _load_tickets()
    ticket_id = f"OPS-{uuid.uuid4().hex[:6].upper()}"
    tickets[ticket_id] = {
        "title": title,
        "description": description,
        "severity": severity,
        "status": "open",
        "created_at": time.time(),
        "updated_at": time.time(),
        "history": [],
    }
    _save_tickets(tickets)
    return f"Created ticket {ticket_id} (severity={severity}, status=open)."


@mcp.tool()
def update_ticket(ticket_id: str, status: Optional[str] = None, note: Optional[str] = None) -> str:
    """
    Update an existing ticket's status and/or append a progress note.

    Args:
        ticket_id: The ticket identifier returned by create_ticket (e.g. "OPS-AB12CD").
        status: New status, e.g. "in_progress", "resolved", "closed". Omit to leave unchanged.
        note: A short note describing what you tried or observed. Omit if there's nothing to add.
    """
    tickets = _load_tickets()
    if ticket_id not in tickets:
        return f"No such ticket: {ticket_id}"

    ticket = tickets[ticket_id]
    if status:
        ticket["status"] = status
    if note:
        ticket["history"].append({"note": note, "at": time.time()})
    ticket["updated_at"] = time.time()

    _save_tickets(tickets)
    return f"Updated {ticket_id}: status={ticket['status']}, notes={len(ticket['history'])}."


@mcp.tool()
def get_dashboard_metrics(service: str) -> str:
    """
    Read current monitoring metrics for a service.

    Use this before deciding whether to act, escalate, or close a
    ticket, and again afterward to confirm whether an action worked.

    Args:
        service: The service/component name as it appears on the dashboard.

    NOTE: placeholder implementation returning simulated metrics.
    Wire this up to a real monitoring backend (Prometheus, Grafana,
    Datadog, CloudWatch, ...) before relying on it operationally.
    """
    # TODO: replace with a real query against your monitoring backend.
    simulated = {
        "service": service,
        "status": "degraded",
        "error_rate_pct": 4.2,
        "p95_latency_ms": 812,
        "cpu_pct": 71,
        "memory_pct": 58,
    }
    return json.dumps(simulated, indent=2)


@mcp.tool()
def escalate_to_human(summary: str, severity: str = "high") -> str:
    """
    Escalate an issue to a human operator - use this when the issue
    is outside the agent's authority to resolve, the risk of acting
    alone is too high (e.g. a production data change), the root
    cause is ambiguous, or a fix has already failed once or twice.

    Args:
        summary: What's wrong, what's been tried so far, and why this needs a human now.
        severity: One of "low", "medium", "high", "critical".

    NOTE: placeholder implementation that logs the escalation
    locally. Wire this up to a real paging/notification channel
    (PagerDuty, Opsgenie, a Slack webhook, ...) before relying on
    it operationally.
    """
    # TODO: replace with a real paging/notification call.
    entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] severity={severity} :: {summary}\n"
    with ESCALATIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(entry)
    return f"Escalated to human operators (severity={severity}). Logged to {ESCALATIONS_LOG.name}."


if __name__ == "__main__":
    mcp.run(transport="stdio")