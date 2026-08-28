"""ARC command-line interface.

Subcommands:
  arc                      interactive chat (default)
  arc chat                 same as above
  arc ask "question"       one-shot question
  arc jobs search|list|analyze|email
  arc email digest|search QUERY
  arc automate run TASK|all   run one automation task (safety: auto mode)
  arc automate start          run the scheduler loop
  arc doctor               health check
  arc init                 create config + profile interactively
  arc skills search QUERY  search the SKILL.md & component library
  arc skills list          list installed skills
  arc skills categories    list skill categories
  arc skills install       clone/install the skill collection
  arc skills add-source    add a catalog source (e.g. awesome-ai-agent-tools)
  arc voice                live voice — always-listening, full duplex, local
  arc voice --once "hi"    one-shot voice query (no mic)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, List, Optional

from . import __version__
from .app import ArcApp, create_app
from .automation.scheduler import Scheduler
from .config import PROJECT_ROOT, load_config
from .profile.profile import Profile

logger = logging.getLogger("arc.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arc",
        description="ARC — personal AI assistant "
                    "(LLM routing, email, internships, computer control).")
    parser.add_argument("--version", action="version",
                        version=f"arc {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose console logging")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat", help="interactive chat (default)")

    ask = subparsers.add_parser("ask", help="one-shot question")
    ask.add_argument("question", nargs="+", help="the question to ask")

    jobs = subparsers.add_parser("jobs", help="internship engine")
    jobs_sub = jobs.add_subparsers(dest="jobs_command", required=True)
    jobs_sub.add_parser("search", help="search boards, score, save")
    jobs_list = jobs_sub.add_parser("list", help="list saved jobs")
    jobs_list.add_argument("--status", default="",
                           help="filter: new/saved/applied/interview/rejected/offer")
    jobs_list.add_argument("--min-score", type=int, default=0)
    jobs_analyze = jobs_sub.add_parser("analyze", help="deep-analyze one job")
    jobs_analyze.add_argument("job_id", type=int)
    jobs_email = jobs_sub.add_parser("email",
                                     help="draft a recruiter email for a job")
    jobs_email.add_argument("job_id", type=int)

    email = subparsers.add_parser("email", help="email engine")
    email_sub = email.add_subparsers(dest="email_command", required=True)
    email_sub.add_parser("digest", help="classify recent mail")
    email_search = email_sub.add_parser("search", help="Gmail search")
    email_search.add_argument("query", nargs="+")

    automate = subparsers.add_parser("automate", help="automation scheduler")
    automate_sub = automate.add_subparsers(dest="automate_command", required=True)
    automate_run = automate_sub.add_parser("run", help="run one task now")
    automate_run.add_argument("task", help="email_check | job_search | "
                                           "deadline_check | all")
    automate_sub.add_parser("start", help="run the scheduler forever")

    subparsers.add_parser("doctor", help="health check")
    subparsers.add_parser("init", help="create config and profile")

    skills = subparsers.add_parser("skills", help="SKILL.md skill library")
    skills_sub = skills.add_subparsers(dest="skills_command", required=True)
    skills_search = skills_sub.add_parser("search", help="search skills")
    skills_search.add_argument("query", nargs="+", help="search terms")
    skills_search.add_argument("--limit", type=int, default=10)
    skills_list = skills_sub.add_parser("list", help="list skills")
    skills_list.add_argument("--category", default="", help="filter by category")
    skills_list.add_argument("--limit", type=int, default=50)
    skills_sub.add_parser("categories", help="list categories with counts")
    skills_install = skills_sub.add_parser("install", help="clone/install the collection")
    skills_install.add_argument("--repo", default="https://github.com/luokai0/ai-agent-skills-by-luo-kai.git",
                                help="source repo URL")
    skills_install.add_argument("--source", default="",
                                help="local copy source dir (skips network clone)")
    skills_add = skills_sub.add_parser("add-source",
                                       help="add a catalog-style source e.g. awesome-ai-agent-tools")
    skills_add.add_argument("--repo", default="https://github.com/michielhdoteth/awesome-ai-agent-tools.git",
                            help="catalog repo URL")
    skills_add.add_argument("--name", default="",
                            help="source name/dir (defaults to repo basename)")
    skills_add.add_argument("--source", default="",
                             help="local copy source dir (skips network clone)")

    voice = subparsers.add_parser("voice", help="live voice — always-listening, full duplex (local RTX 4050)")
    voice.add_argument("--once", nargs="*", default=None,
                       help="one-shot voice query as text (no mic, for testing)")
    voice.add_argument("--stt", default="",
                       help="STT: faster-whisper | fake (default from config)")
    voice.add_argument("--stt-model", default="",
                       help="whisper model: tiny/small/medium/large-v3 (default small)")
    voice.add_argument("--tts", default="",
                       help="TTS: auto | kokoro | piper | pyttsx3 | fake")
    voice.add_argument("--device", default="",
                       help="device: auto | cuda | cpu")
    voice.add_argument("--no-voice-approval", action="store_true",
                       help="disable voice approvals — always use text Confirm")
    return parser


# ------------------------------------------------------------------ commands
def _cmd_chat(app: ArcApp) -> int:
    from .ui.terminal import TerminalUI
    TerminalUI(app).run()
    return 0


def _cmd_ask(app: ArcApp, args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    question = " ".join(args.question)
    console.print(Markdown(app.chat(question)))
    return 0


def _cmd_jobs(app: ArcApp, args: argparse.Namespace) -> int:
    from rich.console import Console
    console = Console()
    engine = app.internships
    if args.jobs_command == "search":
        console.print("[bold]Searching job boards…[/bold]")
        jobs = engine.search(internship_only=True)
        console.print(engine.report(jobs, top=15))
    elif args.jobs_command == "list":
        jobs = app.db.list_jobs(status=args.status or None,
                                min_score=args.min_score, limit=50)
        if not jobs:
            console.print("No saved jobs match. Try `arc jobs search` first.")
            return 0
        for job in jobs:
            console.print(f"#{job['id']} [{job['match_score']}%] {job['role']} — "
                          f"{job['company']} ({job['status']}) {job['location']}")
    elif args.jobs_command == "analyze":
        job = app.db.get_job(args.job_id)
        if job is None:
            console.print(f"No job with id {args.job_id}")
            return 1
        analysis = engine.analyze_job(job)
        console.print(f"[bold]{job['role']} — {job['company']}[/bold]")
        for key in ("requirements", "candidate_strengths", "candidate_gaps"):
            items = analysis.get(key) or []
            if items:
                console.print(f"\n[underline]{key.replace('_', ' ').title()}[/underline]")
                for item in items[:8]:
                    console.print(f"  - {item}")
        if analysis.get("tailored_pitch"):
            console.print(f"\nPitch: {analysis['tailored_pitch']}")
        if analysis.get("error"):
            console.print(f"[red]LLM analysis unavailable: {analysis['error']}[/red]")
    elif args.jobs_command == "email":
        job = app.db.get_job(args.job_id)
        if job is None:
            console.print(f"No job with id {args.job_id}")
            return 1
        draft = engine.draft_recruiter_email(job)
        if draft.get("error"):
            console.print(f"[red]Drafting failed: {draft['error']}[/red]")
            return 1
        console.print(f"[bold]Subject:[/bold] {draft['subject']}\n\n{draft['body']}"
                      "\n\n[dim](Draft only — nothing was sent.)[/dim]")
    return 0


def _cmd_email(app: ArcApp, args: argparse.Namespace) -> int:
    from rich.console import Console
    console = Console()
    problem = app.gmail_engine.availability_message()
    if problem != "ready":
        console.print(f"[yellow]Gmail not ready:[/yellow] {problem}")
        return 1
    if args.email_command == "digest":
        console.print(app.gmail_tools.daily_digest())
    elif args.email_command == "search":
        for message in app.gmail_engine.search(" ".join(args.query)):
            console.print(f"- {message['subject']} — {message['from']}\n"
                          f"  {message['snippet'][:120]}")
    return 0


def _cmd_automate(args: argparse.Namespace) -> int:
    from rich.console import Console
    console = Console()
    # Automation always runs in auto mode: approvals are denied, not prompted.
    config = load_config()
    config.safety_mode = "auto"
    app = ArcApp(config=config)
    scheduler = Scheduler(app)
    try:
        if args.automate_command == "run":
            if args.task == "all":
                results = scheduler.run_all()
                for result in results:
                    mark = "[green]✓[/green]" if result.ok else "[red]✗[/red]"
                    console.print(f"{mark} {result.task} ({result.ran_at}):\n"
                                  f"{result.output}\n")
            else:
                result = scheduler.run_task(args.task)
                console.print(f"{result.output}")
                return 0 if result.ok else 1
        elif args.automate_command == "start":
            console.print("[bold]Scheduler running — Ctrl-C to stop.[/bold]")
            scheduler.run_forever()
    finally:
        app.close()
    return 0


def _cmd_skills(app: ArcApp, args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.markup import escape
    console = Console()
    library = app.skills
    cmd = args.skills_command

    if cmd == "install":
        from .skills.install import install_skills
        from .skills.index import SkillLibrary
        dest = app.skills_root or (app.config.data_dir / "skills")
        console.print(f"[bold]Installing skills into {dest}…[/bold]")
        try:
            root = install_skills(dest, repo_url=args.repo,
                                  source=Path(args.source) if args.source else None)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Install failed:[/red] {exc}")
            return 1
        console.print("[green]✓ Installed.[/green]")
        fresh = SkillLibrary(root)
        console.print(f"  indexed {fresh.count} skills in {fresh.categories_with_counts().__len__()} categories")
        return 0

    if cmd == "add-source":
        from .skills.install import install_source
        from .skills.index import SkillLibrary, SOURCES_DIRNAME
        dest = app.skills_root or (app.config.data_dir / "skills")
        source_root = dest / SOURCES_DIRNAME
        source_root.mkdir(parents=True, exist_ok=True)
        console.print(f"[bold]Adding catalog source into {source_root}…[/bold]")
        try:
            root = install_source(source_root, repo_url=args.repo,
                                  name=args.name,
                                  source=Path(args.source) if args.source else None)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Add-source failed:[/red] {exc}")
            return 1
        console.print(f"[green]✓ Added.[/green]")
        merged = SkillLibrary(dest,
                              extra_roots=[p for p in source_root.iterdir() if p.is_dir()])
        console.print(f"  source now contributes {merged.count} components "
                      f"({merged.categories_with_counts().__len__()} categories)")
        return 0

    if library is None or not library.count:
        console.print("[yellow]Skill library not installed.[/yellow] "
                      "Run `arc skills install` to clone it.")
        return 1

    if cmd == "categories":
        for cat, count in library.categories_with_counts().items():
            console.print(f"- {cat} ({count})")
        console.print(f"\n[dim]{library.count} skills total[/dim]")
        return 0

    if cmd == "search":
        results = library.search(" ".join(args.query), limit=args.limit)
        if not results:
            console.print(f"No skills match {escape(' '.join(args.query))!r}")
            return 1
        for i, entry in enumerate(results, 1):
            kind = f" ({entry.kind})" if entry.kind and not entry.name.endswith(f" ({entry.kind})") else ""
            console.print(f"{i}. [cyan]{entry.name}[/cyan]{escape(kind)} "
                          f"[{entry.category}] — {escape(entry.description)}")
        return 0

    # list
    entries = library.list_by_category(args.category)
    if not entries:
        console.print(f"No skills{ ' in '+escape(args.category) if args.category else '' }.")
        return 1
    for i, entry in enumerate(entries[:args.limit], 1):
        kind = f" ({entry.kind})" if entry.kind and not entry.name.endswith(f" ({entry.kind})") else ""
        console.print(f"{i}. [cyan]{entry.name}[/cyan]{escape(kind)} "
                      f"[{entry.category}] — {escape(entry.description)}")
    if len(entries) > args.limit:
        console.print(f"[dim]… and {len(entries) - args.limit} more "
                      f"(use --limit to show more)[/dim]")
    return 0


def _cmd_doctor(app: ArcApp) -> int:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    report = app.health_report()

    llm = report["llm"]
    if llm["ok"]:
        console.print(f"[green]✓ LLM gateway[/green] {llm['base_url']} — "
                      f"{llm.get('model_count')} models")
        for role, info in llm.get("configured", {}).items():
            mark = "[green]✓[/green]" if info["known"] else "[yellow]✗[/yellow]"
            console.print(f"  model [cyan]{role}[/cyan]: {info['model']} {mark}")
    else:
        console.print(f"[red]✗ LLM gateway[/red] {llm['base_url']} — {llm['error']}")

    console.print(f"  safety mode: {report['safety_mode']}  |  "
                  f"tools: {len(report['tools'])}")

    skills = report.get("skills") or {}
    if skills.get("available"):
        content = "content ✓" if skills.get("content") else "content ✗ (run `arc skills install`)"
        console.print(f"  skills: [green]{skills.get('count')}[/green] indexed "
                      f"({skills.get('categories')} categories) — {content}")
    else:
        console.print("  skills: [yellow]not installed[/yellow] "
                      "(run `arc skills install`)")


    from rich.markup import escape
    table = Table(title="Optional engines")
    table.add_column("tool")
    table.add_column("status", overflow="fold")
    for name, status in report["optional"].items():
        ok = ("not installed" not in status) and ("missing" not in status)
        table.add_row(name, f"[green]{escape(status)}[/green]" if ok
                      else f"[yellow]{escape(status)}[/yellow]")
    console.print(table)
    jobs = report["db"]["jobs"]
    console.print(f"  database: {report['db']['path']} — {jobs['total']} jobs "
                  f"({', '.join(f'{k}={v}' for k, v in jobs.items() if k != 'total')})")
    placeholder = report["profile"]["placeholder"]
    console.print(f"  profile: {report['profile']['path']} — "
                  f"{'[yellow]placeholder[/yellow]' if placeholder else '[green]filled in[/green]'}")
    return 0 if llm["ok"] else 1


def _cmd_voice(app: ArcApp, args: argparse.Namespace) -> int:
    from rich.console import Console
    console = Console()

    # One-shot mode: no mic needed, just handle text via voice pipeline (for testing)
    if args.once is not None:
        text = " ".join(args.once).strip()
        if not text:
            console.print("[yellow]Provide text after --once, e.g. `arc voice --once \"hello\"`[/yellow]")
            return 1
        # One-shot uses FakeSTT/FakeTTS if real deps missing, but still goes through voice session + approval path
        try:
            from .voice import FakeSTT, FakeTTS, VoiceSession
            stt = FakeSTT(script=[text])
            tts = FakeTTS()
            session = VoiceSession(app, stt=stt, tts=tts, allow_voice_approval=not args.no_voice_approval)
            reply = session.handle_once(text)
            console.print(f"[dim]voice once → {reply[:400] if reply else '(no reply)'}[/dim]")
            return 0
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Voice once failed:[/red] {exc}")
            return 1

    # Live mode: always-listening full duplex, local on RTX 4050
    voice_cfg = getattr(app.config, "voice", {}) or {}
    stt_kind = (args.stt or voice_cfg.get("stt") or "faster-whisper").lower()
    stt_model = args.stt_model or voice_cfg.get("stt_model") or "small"
    tts_kind = (args.tts or voice_cfg.get("tts") or "auto").lower()
    device = (args.device or voice_cfg.get("device") or "auto").lower()
    allow_voice_approval = (not args.no_voice_approval) and bool(voice_cfg.get("allow_voice_approval", True))

    # Lazy imports so `arc` still works without voice deps
    try:
        from .voice.stt import FasterWhisperSTT, FakeSTT
        from .voice.tts import make_tts
        from .voice.session import VoiceSession
        from .voice.vad import VADDetector
    except ImportError as exc:
        console.print(f"[red]Voice deps missing:[/red] {exc}")
        console.print("Install with: [cyan]pip install -e \".[voice]\"[/cyan]  (add [kokoro] for Kokoro)")
        console.print("System deps: [cyan]sudo apt install portaudio19-dev espeak espeak-data[/cyan]")
        return 1

    # STT: faster-whisper on CUDA (4050 6GB) — small is 500MB, medium 1.5GB
    if stt_kind == "fake":
        stt = FakeSTT(script=[])
    else:
        try:
            stt = FasterWhisperSTT(model=stt_model, device=device)
            if not stt.available:
                raise RuntimeError("faster-whisper not available")
            console.print(f"[dim]STT: faster-whisper {stt_model} on {stt.device} ({stt.compute_type})[/dim]")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]FasterWhisper failed ({exc}) — using FakeSTT (no mic).[/yellow]")
            stt = FakeSTT(script=[])

    # TTS: auto → kokoro (if VRAM) → piper → pyttsx3
    try:
        tts = make_tts(kind=tts_kind, device=device)
        console.print(f"[dim]TTS: {tts.name} (available={tts.available})[/dim]")
        if not tts.available:
            raise RuntimeError("no TTS available")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]TTS init failed:[/red] {exc}")
        from .voice.tts import FakeTTS as _FakeTTS
        tts = _FakeTTS()

    # Check mic
    try:
        import sounddevice as sd  # type: ignore
        # quick device check
        devs = sd.query_devices()
        console.print(f"[dim]Audio devices: {len(devs)} found[/dim]")
    except Exception:
        console.print("[yellow]sounddevice not available — voice will use text fallback for approvals.[/yellow]")

    vad = VADDetector(aggressiveness=int(voice_cfg.get("vad_aggressiveness", 2)))

    session = VoiceSession(
        app=app,
        stt=stt,
        tts=tts,
        vad=vad,
        console=console,
        allow_voice_approval=allow_voice_approval,
        confidence_threshold=float(voice_cfg.get("confidence_threshold", 0.6)),
    )
    return 0 if session.run_forever() is None else 0  # run_forever handles its own exit


def _cmd_init() -> int:
    from rich.console import Console
    from rich.prompt import Prompt
    console = Console()

    # 1. config.yaml
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    example = PROJECT_ROOT / "config" / "config.example.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.is_file():
        console.print(f"[green]✓[/green] config already exists: {config_path}")
    elif example.is_file():
        shutil.copy(example, config_path)
        console.print(f"[green]✓[/green] created {config_path} (from example)")
    else:
        console.print("[yellow]![/yellow] no config example found — defaults apply")

    # 2. .env
    env_example = PROJECT_ROOT / ".env.example"
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file() and env_example.is_file():
        shutil.copy(env_example, env_path)
        console.print(f"[green]✓[/green] created {env_path} — edit it to set "
                      "ARC_LLM_BASE_URL / ARC_LLM_API_KEY")

    # 3. profile
    config = load_config()
    profile = Profile.load(config.profile_path)
    console.print(f"[green]✓[/green] profile: {config.profile_path}")

    console.print("\n[bold]A few questions[/bold] [dim](Enter keeps the current "
                  "value)[/dim]")
    updates = {
        "name": Prompt.ask("Your name", default=profile.name),
        "email": Prompt.ask("Email", default=str(profile.get("email", ""))),
        "location": Prompt.ask("Location", default=str(profile.get("location", ""))),
    }
    roles = Prompt.ask("Preferred roles (comma separated)",
                       default=", ".join(str(r) for r in
                                         profile.get("preferred_roles") or []))
    changed = False
    for key, value in updates.items():
        if value and value != profile.get(key):
            profile[key] = value
            changed = True
    if roles:
        role_list = [r.strip() for r in roles.split(",") if r.strip()]
        if role_list != [str(r) for r in profile.get("preferred_roles") or []]:
            profile["preferred_roles"] = role_list
            changed = True
    if changed:
        profile.save()
        console.print("[green]✓[/green] profile updated")
    else:
        console.print("[dim]no changes[/dim]")

    console.print("\nNext steps:")
    console.print("  arc doctor    — verify your setup")
    console.print("  arc           — start chatting")
    console.print(f"  edit {config.profile_path} — full resume, skills, projects")
    return 0


# --------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "chat"

    if command == "init":
        return _cmd_init()
    if command == "automate":
        return _cmd_automate(args)

    app = create_app()
    if args.verbose:
        for handler in logging.getLogger("arc").handlers:
            handler.setLevel(logging.DEBUG)
    try:
        if command == "chat":
            return _cmd_chat(app)
        if command == "ask":
            return _cmd_ask(app, args)
        if command == "jobs":
            return _cmd_jobs(app, args)
        if command == "email":
            return _cmd_email(app, args)
        if command == "skills":
            return _cmd_skills(app, args)
        if command == "voice":
            return _cmd_voice(app, args)
        if command == "doctor":
            return _cmd_doctor(app)
        parser.error(f"Unknown command: {command}")
        return 2
    except KeyboardInterrupt:
        print()
        return 130
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
