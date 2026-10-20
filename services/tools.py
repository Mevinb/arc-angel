"""
Typed tool registry with capability manifests, policy enforcement, budgets,
retries and audit logging.

Every agent action is a named tool with a JSON-schema contract.  Handlers never
receive raw credentials; policy checks happen before execution and all inputs /
outputs are redacted before they are persisted to the audit trail.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import DATA_DIR, WORKSPACES_DIR
from services.audit import AuditService
from services.crypto import looks_like_secret, redact_obj, redact_text
from services.policies import Decision, PolicyService, RiskClass, SANDBOX_DIR, args_hash
from services.safety import SafetyService

ToolHandler = Callable[["ToolContext", Dict[str, Any]], Awaitable[Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolValidationError(ValueError):
    pass


@dataclass
class ToolSpec:
    name: str
    title: str
    description: str
    schema: Dict[str, Any]
    risk_class: str = RiskClass.READ_ONLY
    handler: Optional[ToolHandler] = None
    network: bool = False
    side_effects: bool = False
    idempotent: bool = True
    max_retries: int = 1
    timeout_seconds: float = 60.0
    requires_workspace: bool = False
    hidden: bool = False
    tags: List[str] = field(default_factory=list)
    plugin_id: Optional[str] = None

    def manifest(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "input_schema": self.schema,
            "risk_class": self.risk_class,
            "capabilities": {
                "network": self.network,
                "side_effects": self.side_effects,
                "idempotent": self.idempotent,
                "requires_workspace": self.requires_workspace,
            },
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
            "plugin_id": self.plugin_id,
        }


@dataclass
class ToolContext:
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    run_id: Optional[str] = None
    workspace_dir: Optional[str] = None
    policy_id: Optional[str] = None
    autopilot: bool = False
    interactive: bool = True
    approval_id: Optional[str] = None
    simulate: bool = False
    budget: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    connector_id: Optional[str] = None
    domain: Optional[str] = None
    contact: Optional[str] = None
    path: Optional[str] = None
    external_roots: List[str] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return int(self.usage.get("tool_calls", 0) or 0)

    @property
    def max_tool_calls(self) -> Optional[int]:
        value = self.budget.get("max_tool_calls")
        return int(value) if value is not None else None

    def workspace_root(self) -> Path:
        if self.workspace_dir:
            base = Path(self.workspace_dir).expanduser().resolve()
        else:
            key = self.run_id or self.agent_id or "default"
            base = (SANDBOX_DIR / str(key)).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base

    def policy_context(self, tool: ToolSpec, args: Dict[str, Any]) -> Dict[str, Any]:
        path = self.path
        if not path:
            for key in ("path", "file", "src", "source", "target"):
                if isinstance(args.get(key), str):
                    path = args[key]
                    break
        domain = self.domain
        if not domain:
            for key in ("url", "endpoint"):
                value = args.get(key)
                if isinstance(value, str) and "://" in value:
                    from urllib.parse import urlparse

                    domain = urlparse(value).hostname
                    break
        contact = self.contact or args.get("contact") or args.get("to") or args.get("recipient")
        return {
            "action": f"tool:{tool.name}",
            "tool_name": tool.name,
            "risk_class": tool.risk_class,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "connector_id": self.connector_id,
            "domain": domain,
            "contact": contact,
            "path": path,
            "autopilot": self.autopilot,
            "interactive": self.interactive,
            "amount_usd": args.get("amount_usd") or args.get("amount"),
            "policy_id": self.policy_id,
        }

    def safe_path(self, rel_path: str) -> Path:
        base = self.workspace_root()
        raw = str(rel_path or ".").strip()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = base / candidate
        target = candidate.expanduser().resolve()
        allowed = target == base or base in target.parents
        if not allowed and self.external_roots:
            for root in self.external_roots:
                try:
                    r = Path(root).expanduser().resolve()
                    if target == r or r in target.parents:
                        allowed = True
                        break
                except Exception:
                    continue
        if not allowed:
            raise PermissionError(
                f"Path '{rel_path}' escapes the approved workspace. "
                "External folders need an explicit folder policy grant."
            )
        return target


def _validate_value(value: Any, schema: Dict[str, Any], path: str) -> Any:
    expected = schema.get("type")
    if expected == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path} must be a string")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            value = value[: int(schema["maxLength"])]
        if "enum" in schema and value not in schema["enum"]:
            raise ToolValidationError(f"{path} must be one of {schema['enum']}")
    elif expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ToolValidationError(f"{path} must be an integer")
        if "minimum" in schema and value < int(schema["minimum"]):
            raise ToolValidationError(f"{path} must be >= {schema['minimum']}")
        if "maximum" in schema and value > int(schema["maximum"]):
            raise ToolValidationError(f"{path} must be <= {schema['maximum']}")
    elif expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolValidationError(f"{path} must be a number")
    elif expected == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path} must be a boolean")
    elif expected == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path} must be an array")
        item_schema = schema.get("items")
        if item_schema:
            value = [_validate_value(v, item_schema, f"{path}[]") for v in value]
    elif expected == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{path} must be an object")
    return value


def validate_args(schema: Dict[str, Any], args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolValidationError("Tool arguments must be an object")
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    cleaned = dict(args)
    for key in required:
        if key not in cleaned:
            raise ToolValidationError(f"Missing required argument: {key}")
    for key, sub_schema in properties.items():
        if key in cleaned and cleaned[key] is not None:
            cleaned[key] = _validate_value(cleaned[key], sub_schema, key)
        elif "default" in sub_schema and key not in cleaned:
            cleaned[key] = sub_schema["default"]
    return cleaned


class ToolRegistry:
    _instance: Optional["ToolRegistry"] = None

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._builtins_loaded = False
        self._plugin_loaded = False
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = ToolRegistry()
        return cls._instance

    def register(self, spec: ToolSpec, replace: bool = False) -> None:
        if spec.name in self._tools and not replace:
            return
        if not spec.handler:
            raise ValueError("Tool handler is required")
        self._tools[spec.name] = spec

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        self._ensure_builtins()
        return self._tools.get(name)

    def list_tools(self, include_hidden: bool = False) -> List[Dict[str, Any]]:
        self._ensure_builtins()
        specs = [s for s in self._tools.values() if include_hidden or not s.hidden]
        return [s.manifest() for s in sorted(specs, key=lambda s: s.name)]

    def _ensure_builtins(self) -> None:
        if not self._builtins_loaded:
            from services.builtin_tools import register_builtin_tools

            self._builtins_loaded = True
            register_builtin_tools(self)
        if not self._plugin_loaded:
            self._plugin_loaded = True
            try:
                self.reload_plugin_tools()
            except Exception:
                pass

    def reload_plugin_tools(self) -> int:
        """Load tools exposed by installed + enabled local plugins."""
        for name in [n for n, s in self._tools.items() if s.plugin_id]:
            self._tools.pop(name, None)
        count = 0
        try:
            from services.plugins import PluginService

            for plugin in PluginService.list_plugins(enabled_only=True):
                manifest = plugin.get("manifest") or {}
                for tool in manifest.get("tools", []):
                    if not isinstance(tool, dict) or not tool.get("name"):
                        continue
                    tool_name = f"plugin.{plugin['id']}.{tool['name']}".lower().replace(" ", "_")
                    endpoint = tool.get("endpoint") or manifest.get("endpoint")
                    if not endpoint:
                        continue
                    risk = tool.get("risk_class", RiskClass.REVERSIBLE_WRITE)
                    if risk not in RiskClass.ALL:
                        risk = RiskClass.RESTRICTED
                    spec = ToolSpec(
                        name=tool_name,
                        title=tool.get("title") or tool["name"],
                        description=tool.get("description") or f"Plugin tool from {plugin['name']}",
                        schema=tool.get("input_schema") or {"type": "object", "properties": {}},
                        risk_class=risk,
                        handler=self._plugin_handler(tool_name, endpoint),
                        network=str(endpoint).startswith("http"),
                        side_effects=True,
                        idempotent=False,
                        timeout_seconds=45.0,
                        tags=["plugin", plugin["id"]],
                        plugin_id=plugin["id"],
                    )
                    self.register(spec, replace=True)
                    count += 1
        except Exception:
            pass
        return count

    @staticmethod
    def _plugin_handler(tool_name: str, endpoint: str) -> ToolHandler:
        async def handler(ctx: ToolContext, args: Dict[str, Any]) -> Any:
            import httpx

            if ctx.simulate:
                return {"simulated": True, "would_call": endpoint, "arguments": redact_obj(args)}
            async with httpx.AsyncClient(timeout=40.0) as client:
                res = await client.post(endpoint, json={"tool": tool_name, "arguments": args}, headers={"X-Grok-Plugin": "1"})
            body = res.text[:20000]
            return {"status_code": res.status_code, "body": body, "plugin_tool": tool_name}

        return handler

    async def execute(
        self,
        name: str,
        args: Optional[Dict[str, Any]],
        ctx: Optional[ToolContext] = None,
        *,
        approval_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_builtins()
        spec = self._tools.get(name)
        if not spec:
            return {"status": "error", "tool": name, "error": f"Unknown tool: {name}"}
        ctx = ctx or ToolContext()
        if approval_id and not ctx.approval_id:
            ctx.approval_id = approval_id

        started = time.time()
        try:
            cleaned = validate_args(spec.schema, args)
        except ToolValidationError as exc:
            return {"status": "error", "tool": name, "error": str(exc)}

        if SafetyService.kill_switch_engaged():
            result = {
                "status": "denied", "tool": name,
                "error": "Global kill switch is engaged. Release it in Command Center → Safety.",
            }
            AuditService.log(
                "tool", f"tool.{name}", actor="system", status="denied",
                risk_class=spec.risk_class, tool_name=name, agent_id=ctx.agent_id,
                run_id=ctx.run_id, task_id=ctx.task_id, summary=result["error"],
                data={"args": cleaned}, activity_title=f"Tool denied: {name}", severity="warning",
            )
            return result

        if spec.network and SafetyService.offline_mode():
            result = {"status": "denied", "tool": name, "error": "Offline mode is enabled; network tools are unavailable."}
            AuditService.log(
                "tool", f"tool.{name}", actor="system", status="denied",
                risk_class=spec.risk_class, tool_name=name, agent_id=ctx.agent_id,
                run_id=ctx.run_id, task_id=ctx.task_id, summary=result["error"],
                data={"args": cleaned}, activity_title=f"Tool denied: {name}", severity="warning",
            )
            return result

        if ctx.max_tool_calls is not None and ctx.tool_calls >= ctx.max_tool_calls:
            return {"status": "denied", "tool": name, "error": "Run tool-call budget exhausted."}

        policy_ctx = ctx.policy_context(spec, cleaned)
        decision = PolicyService.evaluate(policy_ctx)

        if ctx.approval_id:
            approval = PolicyService.get_approval(ctx.approval_id)
            if not approval or approval["status"] != "approved":
                return {"status": "denied", "tool": name, "error": "Approval is missing or not approved."}
            if approval.get("tool_name") and approval["tool_name"] != name:
                return {"status": "denied", "tool": name, "error": "Approval does not match this tool."}
            if approval.get("run_id") and ctx.run_id and approval["run_id"] != ctx.run_id:
                return {"status": "denied", "tool": name, "error": "Approval does not match this run."}
            expected_hash = (approval.get("payload") or {}).get("args_hash")
            if expected_hash and expected_hash != args_hash(cleaned):
                return {"status": "denied", "tool": name, "error": "Approval does not match the requested arguments."}
            if decision["decision"] == Decision.DENY:
                return {
                    "status": "denied", "tool": name,
                    "error": "; ".join(decision.get("reasons") or ["Denied by policy"]),
                    "risk_class": spec.risk_class,
                }
        elif decision["decision"] == Decision.REQUIRE_APPROVAL:
            approval = PolicyService.request_approval(
                policy_ctx,
                reason="; ".join(decision.get("reasons") or ["Approval required"]),
                payload={"args": cleaned, "tool": spec.manifest(), "context": policy_ctx},
            )
            return {
                "status": "awaiting_approval",
                "tool": name,
                "approval_id": approval["id"],
                "risk_class": spec.risk_class,
                "reasons": decision.get("reasons"),
                "message": "This action is paused until you approve it in the approval inbox.",
            }
        elif decision["decision"] == Decision.DENY:
            result = {
                "status": "denied", "tool": name,
                "error": "; ".join(decision.get("reasons") or ["Denied by policy"]),
                "risk_class": spec.risk_class,
            }
            AuditService.log(
                "tool", f"tool.{name}", actor="agent" if ctx.agent_id else "system",
                status="denied", risk_class=spec.risk_class, tool_name=name,
                agent_id=ctx.agent_id, run_id=ctx.run_id, task_id=ctx.task_id,
                summary=result["error"], data={"args": redact_obj(cleaned), "decision": decision},
                activity_title=f"Tool denied: {name}", severity="warning",
            )
            return result

        if ctx.simulate and not spec.side_effects:
            # Read-only tools still execute in simulation so previews are useful.
            pass

        output: Any = None
        error: Optional[str] = None
        attempts = 0
        max_attempts = 1 + (spec.max_retries if spec.idempotent else 0)
        while attempts < max_attempts:
            attempts += 1
            try:
                output = await asyncio.wait_for(spec.handler(ctx, cleaned), timeout=spec.timeout_seconds)
                if isinstance(output, dict) and output.get("success") is False:
                    error = str(output.get("error") or "Tool reported failure")
                    break
                if (
                    isinstance(output, dict)
                    and isinstance(output.get("status_code"), int)
                    and output["status_code"] >= 400
                ):
                    error = f"HTTP {output['status_code']}"
                    break
                error = None
                break
            except asyncio.TimeoutError:
                error = f"Tool timed out after {spec.timeout_seconds:.0f}s"
                if not spec.idempotent:
                    break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if not spec.idempotent or attempts >= max_attempts:
                    break
                await asyncio.sleep(min(1.0, 0.15 * attempts))

        duration_ms = round((time.time() - started) * 1000, 1)
        ctx.usage["tool_calls"] = ctx.tool_calls + 1

        safe_args = redact_obj(cleaned)
        safe_output = redact_obj(output)
        status = "error" if error else "ok"
        if ctx.simulate:
            status = "simulated" if not error else "error"
        summary = f"{spec.title} {'simulated' if ctx.simulate else ('failed' if error else 'completed')}"
        if error:
            summary = f"{spec.title} failed: {error}"
        event = AuditService.log(
            "tool", f"tool.{name}", actor="agent" if ctx.agent_id else "user",
            status=status, risk_class=spec.risk_class, tool_name=name,
            agent_id=ctx.agent_id, run_id=ctx.run_id, task_id=ctx.task_id,
            approval_id=ctx.approval_id, summary=summary,
            data={
                "input": safe_args, "output": safe_output, "error": error,
                "duration_ms": duration_ms, "attempts": attempts,
                "simulated": ctx.simulate, "policy_decision": decision,
            },
            activity_title=f"{spec.title} {status}", severity="info" if status == "ok" else "warning",
        )
        result = {
            "status": status,
            "tool": name,
            "output": safe_output,
            "duration_ms": duration_ms,
            "audit_id": event["id"],
            "simulated": ctx.simulate,
            "risk_class": spec.risk_class,
        }
        if error:
            result["error"] = error
        return result


def get_tool_registry() -> ToolRegistry:
    return ToolRegistry.get()
