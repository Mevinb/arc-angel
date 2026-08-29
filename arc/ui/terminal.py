"""Phase 10 — Terminal UI: a rich-based chat REPL.

Features:
- Markdown-rendered assistant replies with a spinner while the agent thinks.
- Live tool-activity lines ("→ web.fetch …").
- Human approval prompts for YELLOW/RED actions (wired into the PermissionGuard).
- Slash commands: /help /tools /new /doctor /stats /mode /quit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from ..app import ArcApp
from ..safety.permissions import Action, RiskLevel

logger = logging.getLogger("arc.ui")

BANNER = r"""
         █████╗ ██████╗  ██████╗
        ██╔══██╗██╔══██╗██╔════╝
        ███████║██████╔╝██║
        ██╔══██║██╔══██╗██║
        ██║  ██║██║  ██║╚██████╗
        ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
"""

RISK_STYLE = {
    RiskLevel.GREEN: "green",
    RiskLevel.YELLOW: "yellow",
    RiskLevel.RED: "red",
}

HELP_TEXT = """\
Commands:
  /help          this help
  /tools         list available tools with risk levels
  /new           start a fresh conversation (keeps long-term memory)
  /doctor        run health checks (LLM, tools, database)
  /stats         token usage and permission decisions
  /mode [mode]   show or set safety mode (interactive / auto / yolo)
  /quit          exit (also Ctrl-D)
Anything else is sent to the assistant."""


class TerminalUI:
    """The interactive chat loop."""

    def __init__(self, app: ArcApp) -> None:
        self.app = app
        self.console = Console()
        self.app.set_approver(self._approve)
        self.app.orchestrator.on_event = self._on_event

    # ------------------------------------------------------------- approvals
    def _approve(self, action: Action) -> bool:
        # Pause the "thinking…" spinner so the interactive prompt is not fighting
        # it for the console region (otherwise the y/N prompt is hidden and the
        # action silently reads as denied).
        status = getattr(self, "_status", None)
        if status is not None:
            status.stop()
        style = RISK_STYLE.get(action.risk, "white")
        self.console.print()
        self.console.print(Panel(
            Text.assemble(
                (f"[{action.risk.value.upper()}] ", style),
                (action.description, "bold"),
                (f"\n\n{action.details[:500]}" if action.details else "", "dim"),
            ),
            title="Approval required", border_style=style))
        try:
            # Offer task/session-scoped approvals so user isn't asked per-cmd
            choice = Prompt.ask(
                "Allow this?",
                choices=["y", "n", "a", "s", "yes", "no", "all", "session"],
                default="n",
                show_choices=False,
                console=self.console,
            ).strip().lower()
            if choice in ("y", "yes"):
                return True
            if choice in ("a", "all", "always"):
                # Approve all YELLOW (and RED if this was RED) for this task
                try:
                    self.app.guard.approve_for_task(action.risk, scope="task")
                except Exception:
                    pass
                self.console.print("[dim]Approved for this task — won't ask again for similar actions.[/dim]")
                return True
            if choice in ("s", "session"):
                try:
                    self.app.guard.approve_for_task(action.risk, scope="session")
                except Exception:
                    pass
                self.console.print("[dim]Approved for session — won't ask again this session.[/dim]")
                return True
            return False
        except (KeyboardInterrupt, EOFError):
            return False
        finally:
            if status is not None:
                status.start()

    # ---------------------------------------------------------------- events
    def _on_event(self, event: str, data: Dict[str, Any]) -> None:
        if event == "tool_result":
            mark = "[green]✓[/green]" if data.get("ok") else "[red]✗[/red]"
            self.console.print(f"  {mark} {data.get('tool', '?')}")

    # ------------------------------------------------------------------- run
    def run(self) -> None:
        console = self.console
        console.print(Text(BANNER, style="cyan bold"))
        console.print(f"  ARC ready — safety mode: "
                      f"[bold]{self.app.guard.mode}[/bold]. "
                      "Type /help for commands.\n")
        if self.app.profile.is_placeholder():
            console.print("[dim]Tip: run `arc init` to fill in your profile "
                          "for better internship matching.[/dim]\n")

        while True:
            try:
                user_input = console.input("[bold cyan]you ›[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                console.print("Goodbye!")
                break
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    break
                continue

            console.print()
            status = console.status("[bold blue]arc is thinking…[/bold blue]")
            self._status = status
            status.start()
            try:
                turn = self.app.orchestrator.handle(user_input)
            except Exception as exc:  # noqa: BLE001 - UI must never crash
                logger.exception("Agent turn failed")
                turn = None
                console.print(Panel(f"The agent crashed: {exc}",
                                    title="error", border_style="red"))
            finally:
                status.stop()
                self._status = None
            if turn is not None:
                console.print(Panel(Markdown(turn.reply or "(empty)"),
                                    title="arc", border_style="cyan"))
                bits = []
                if turn.model:
                    bits.append(f"model: {turn.model}")
                if turn.tool_calls:
                    bits.append(f"{turn.tool_calls} tool call(s), "
                                f"{turn.iterations} step(s)")
                if bits:
                    console.print(f"[dim]{'  ·  '.join(bits)}[/dim]")
            console.print()

    # -------------------------------------------------------------- commands
    def _handle_command(self, line: str) -> bool:
        """Handle a /command. Returns True when the UI should exit."""
        console = self.console
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command == "/help":
            console.print(HELP_TEXT)
        elif command == "/tools":
            table = Table(title="Tools")
            table.add_column("name", style="cyan")
            table.add_column("risk")
            table.add_column("description", overflow="fold")
            for tool in sorted(self.app.registry.all_tools(),
                               key=lambda t: t.name):
                table.add_row(tool.name,
                              f"[{RISK_STYLE[tool.risk]}]{tool.risk.value}[/]",
                              tool.description)
            console.print(table)
        elif command == "/new":
            self.app.orchestrator.reset()
            console.print("[dim]Conversation cleared.[/dim]")
        elif command == "/doctor":
            self._doctor()
        elif command == "/stats":
            usage = self.app.router.total_usage
            console.print(f"Tokens — prompt: {usage['prompt_tokens']:,}, "
                          f"completion: {usage['completion_tokens']:,}")
            console.print(f"Permissions — {self.app.guard.stats()}")
        elif command == "/mode":
            if argument in ("interactive", "auto", "yolo"):
                if argument == "yolo":
                    if not Confirm.ask("YOLO disables ALL approvals. Really?",
                                       default=False, console=console):
                        return False
                self.app.guard.mode = argument
                console.print(f"Safety mode set to [bold]{argument}[/bold].")
            else:
                console.print(f"Current mode: [bold]{self.app.guard.mode}[/bold] "
                              "(interactive / auto / yolo)")
        elif command in ("/quit", "/exit"):
            return True
        else:
            console.print(f"[yellow]Unknown command {command}. Try /help.[/yellow]")
        return False

    def _doctor(self) -> None:
        console = self.console
        report = self.app.health_report()
        llm = report["llm"]
        if llm["ok"]:
            console.print(f"[green]✓ LLM gateway[/green] {llm['base_url']} — "
                          f"{llm.get('model_count')} models")
            for role, info in llm.get("configured", {}).items():
                mark = "[green]✓[/green]" if info["known"] else "[yellow]✗[/yellow]"
                console.print(f"  model [cyan]{role}[/cyan]: {info['model']} {mark}")
        else:
            console.print(f"[red]✗ LLM gateway[/red] {llm['base_url']} — {llm['error']}")
        console.print(f"  safety mode: {report['safety_mode']}, "
                      f"tools: {len(report['tools'])}")
        for name, status in report["optional"].items():
            mark = "[green]✓[/green]" if "not installed" not in status and \
                "missing" not in status else "[yellow]![/yellow]"
            console.print(f"  {mark} {name}: {escape(status)}")
        jobs = report["db"]["jobs"]
        console.print(f"  database: {report['db']['path']} — {jobs}")


__all__ = ["TerminalUI"]
