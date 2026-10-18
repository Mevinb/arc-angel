"""
Process Manager for llama-server lifecycle (Start, Stop, Health Check, Auto-GPU tuning).
Provides zero-zombie port cleanup, HTTP 503 model-loading tolerance, dynamic VRAM re-checks,
and automatic fallback to reduced GPU layers or CPU mode on Out-Of-Memory (OOM).
"""
import os
import signal
import subprocess
import time
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List
from pathlib import Path
from config import (
    LLAMA_SERVER_BIN,
    LLAMA_HOST,
    LLAMA_PORT,
    DEFAULT_CTX_SIZE,
    DATA_DIR,
    MODELS,
    DEFAULT_MODEL_ID,
    get_active_model_id,
    set_active_model_id,
    get_active_model_config,
    detect_recommended_gpu_layers,
    FLASH_ATTN
)

PID_FILE = DATA_DIR / "llama_server.pid"
LOG_FILE = DATA_DIR / "llama_server.log"
ACTIVE_PROCESS_INFO = DATA_DIR / "running_process.json"

OOM_ERROR_KEYWORDS = [
    "out of memory",
    "cuda error",
    "ggml_cuda_host_malloc",
    "failed to allocate",
    "cudamalloc failed",
    "cuda driver",
    "insufficient vram",
    "all offloaded layers do not fit"
]


class LlamaProcessManager:
    _last_status_message: str = "Engine offline"
    _is_loading: bool = False
    _fallback_occurred: bool = False

    @staticmethod
    def get_running_pid() -> Optional[int]:
        """Returns the PID of the active llama-server process if running."""
        if not PID_FILE.exists():
            return None
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, OSError):
            if PID_FILE.exists():
                PID_FILE.unlink()
            if ACTIVE_PROCESS_INFO.exists():
                ACTIVE_PROCESS_INFO.unlink()
            return None

    @classmethod
    def is_running(cls) -> bool:
        return cls.get_running_pid() is not None

    @classmethod
    def get_recent_log_snippet(cls, max_lines: int = 25) -> str:
        """Reads the trailing lines from the llama-server log file."""
        if not LOG_FILE.exists():
            return ""
        try:
            lines = LOG_FILE.read_text(errors="replace").splitlines()
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""

    @classmethod
    def detect_oom_or_cuda_error(cls) -> bool:
        """Checks if recent log output contains GPU out-of-memory or CUDA allocation failures."""
        snippet = cls.get_recent_log_snippet(40).lower()
        return any(kw in snippet for kw in OOM_ERROR_KEYWORDS)

    @classmethod
    def cleanup_port(cls, port: int = LLAMA_PORT) -> None:
        """Terminates any stale/zombie process holding the llama-server port."""
        pids: List[int] = []
        try:
            res = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2
            )
            pids = [int(p.strip()) for p in res.stdout.strip().split() if p.strip().isdigit()]
        except Exception:
            pids = []

        if not pids:
            try:
                res = subprocess.run(
                    ["fuser", f"{port}/tcp"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=2
                )
                pids = [int(p.strip()) for p in res.stdout.strip().split() if p.strip().isdigit()]
            except Exception:
                pids = []

        my_pid = os.getpid()
        target_pids = [p for p in pids if p != my_pid]

        if target_pids:
            for pid in target_pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
            time.sleep(0.8)
            for pid in target_pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    @classmethod
    def _spawn_process(
        cls,
        model_path: Path,
        mmproj_path: Optional[Path],
        alias: str,
        ctx_size: int,
        ngl: int,
        threads: int
    ) -> subprocess.Popen:
        """Launches llama-server subprocess."""
        cmd = [
            str(LLAMA_SERVER_BIN),
            "-m", str(model_path),
            "--host", LLAMA_HOST,
            "--port", str(LLAMA_PORT),
            "-c", str(ctx_size),
            "-ngl", str(ngl),
            "-t", str(threads),
            "-fa", str(FLASH_ATTN),
            "--split-mode", "none",
            "--alias", alias
        ]

        if mmproj_path and Path(mmproj_path).exists():
            cmd.extend(["--mmproj", str(mmproj_path), "--mmproj-offload", "--image-min-tokens", "1024"])

        log_fd = open(LOG_FILE, "w")
        return subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setpgrp
        )

    @classmethod
    def start(
        cls,
        model_id: Optional[str] = None,
        gpu_layers: Optional[int] = None,
        ctx_size: int = DEFAULT_CTX_SIZE,
        threads: int = 8
    ) -> Dict[str, Any]:
        """
        Starts llama-server with configured model, mmproj, GPU layers, and context size.
        Features zero-zombie port cleanup and automatic OOM layer fallback.
        """
        if model_id and model_id in MODELS:
            set_active_model_id(model_id)

        model_cfg = get_active_model_config()
        model_path = model_cfg["path"]
        mmproj_path = model_cfg.get("mmproj")

        # Check if already running the requested model
        if cls.is_running():
            running_info = {}
            if ACTIVE_PROCESS_INFO.exists():
                try:
                    running_info = json.loads(ACTIVE_PROCESS_INFO.read_text())
                except Exception:
                    pass
            if running_info.get("model_id") == model_cfg["id"]:
                return {
                    "status": "already_running",
                    "pid": cls.get_running_pid(),
                    "model_id": model_cfg["id"]
                }
            cls.stop()

        if not LLAMA_SERVER_BIN.exists():
            cls._last_status_message = f"Binary not found: {LLAMA_SERVER_BIN}"
            return {"status": "error", "message": cls._last_status_message}
        if not model_path.exists():
            cls._last_status_message = f"Model not found: {model_path}"
            return {"status": "error", "message": cls._last_status_message}

        # Clean up any zombie processes holding the port before attempting startup
        cls.cleanup_port(LLAMA_PORT)

        # Dynamic real-time VRAM re-check
        initial_ngl = gpu_layers if gpu_layers is not None else detect_recommended_gpu_layers(model_cfg["id"])
        ngl_candidates = [initial_ngl]

        if initial_ngl > 0:
            half_ngl = max(0, initial_ngl // 2)
            if half_ngl not in ngl_candidates:
                ngl_candidates.append(half_ngl)
            if 0 not in ngl_candidates:
                ngl_candidates.append(0)  # CPU fallback

        proc = None
        chosen_ngl = initial_ngl
        cls._fallback_occurred = False

        for candidate_ngl in ngl_candidates:
            chosen_ngl = candidate_ngl
            if chosen_ngl < initial_ngl:
                cls._fallback_occurred = True
                cls.cleanup_port(LLAMA_PORT)
                time.sleep(0.5)

            proc = cls._spawn_process(
                model_path=model_path,
                mmproj_path=mmproj_path,
                alias=model_cfg.get("alias", "llama-model"),
                ctx_size=ctx_size,
                ngl=chosen_ngl,
                threads=threads
            )

            # Monitor for immediate startup crash / OOM (within first 2.5 seconds)
            time.sleep(2.5)
            if proc.poll() is None:
                # Process survived initial load!
                break

            # If process crashed, check if it was an OOM
            if cls.detect_oom_or_cuda_error():
                continue
            else:
                # Other non-OOM error; no point retrying with lower layers
                break

        if not proc or proc.poll() is not None:
            err_snippet = cls.get_recent_log_snippet(15)
            cls._last_status_message = f"Server failed to start: {err_snippet[-150:] if err_snippet else 'Unknown error'}"
            return {"status": "error", "message": cls._last_status_message}

        PID_FILE.write_text(str(proc.pid))
        ACTIVE_PROCESS_INFO.write_text(json.dumps({
            "pid": proc.pid,
            "model_id": model_cfg["id"],
            "model_name": model_cfg["name"],
            "has_vision": bool(mmproj_path and Path(mmproj_path).exists()),
            "gpu_layers": chosen_ngl,
            "ctx_size": ctx_size,
            "fallback": cls._fallback_occurred
        }))

        if cls._fallback_occurred:
            cls._last_status_message = f"Started with reduced GPU offload ({chosen_ngl} layers) due to low VRAM"
        else:
            cls._last_status_message = "Loading model weights into VRAM..."

        cls._is_loading = True

        return {
            "status": "started",
            "pid": proc.pid,
            "model_id": model_cfg["id"],
            "model_name": model_cfg["name"],
            "has_vision": bool(mmproj_path and Path(mmproj_path).exists()),
            "gpu_layers": chosen_ngl,
            "ctx_size": ctx_size,
            "fallback": cls._fallback_occurred,
            "status_message": cls._last_status_message,
            "log_file": str(LOG_FILE)
        }

    @classmethod
    def stop(cls) -> Dict[str, Any]:
        """Stops the llama-server process."""
        pid = cls.get_running_pid()
        if not pid:
            cls.cleanup_port(LLAMA_PORT)
            cls._last_status_message = "Engine stopped"
            cls._is_loading = False
            return {"status": "not_running"}

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(1.0)
        except OSError:
            pass

        # Ensure port is clean
        cls.cleanup_port(LLAMA_PORT)

        if PID_FILE.exists():
            PID_FILE.unlink()
        if ACTIVE_PROCESS_INFO.exists():
            ACTIVE_PROCESS_INFO.unlink()

        cls._last_status_message = "Engine stopped"
        cls._is_loading = False
        return {"status": "stopped", "pid": pid}

    @classmethod
    def switch_model(cls, model_id: str, gpu_layers: Optional[int] = None, ctx_size: int = DEFAULT_CTX_SIZE) -> Dict[str, Any]:
        """Stops current model (if running) and starts the specified model."""
        cls.stop()
        time.sleep(1.0)
        return cls.start(model_id=model_id, gpu_layers=gpu_layers, ctx_size=ctx_size)

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        """Returns status, loading state, and resource info."""
        pid = cls.get_running_pid()
        model_cfg = get_active_model_config()
        mmproj_path = model_cfg.get("mmproj")
        return {
            "running": pid is not None,
            "is_loading": cls._is_loading,
            "status_message": cls._last_status_message,
            "fallback_occurred": cls._fallback_occurred,
            "pid": pid,
            "active_model_id": model_cfg["id"],
            "active_model_name": model_cfg["name"],
            "model": model_cfg["path"].name,
            "has_vision": bool(mmproj_path and Path(mmproj_path).exists()),
            "recommended_ngl": detect_recommended_gpu_layers(model_cfg["id"]),
            "default_ngl": model_cfg.get("default_ngl", 24),
            "total_layers": model_cfg.get("total_layers", 32),
            "log_file": str(LOG_FILE) if LOG_FILE.exists() else None
        }

    @classmethod
    def ensure_running(
        cls,
        model_id: Optional[str] = None,
        gpu_layers: Optional[int] = None,
        ctx_size: int = DEFAULT_CTX_SIZE,
        wait_seconds: float = 60.0
    ) -> bool:
        """
        Ensures llama-server is running and responsive. Handles HTTP 503 model-loading
        progress and fast-fails if the subprocess crashes.
        """
        if not cls.is_running():
            res = cls.start(model_id=model_id, gpu_layers=gpu_layers, ctx_size=ctx_size)
            if res.get("status") == "error":
                return False

        start_t = time.time()
        while time.time() - start_t < wait_seconds:
            pid = cls.get_running_pid()
            if not pid:
                # Subprocess exited/died - fail fast!
                err = cls.get_recent_log_snippet(15)
                cls._last_status_message = f"Process exited: {err[-120:] if err else 'Unknown error'}"
                cls._is_loading = False
                return False

            try:
                req = urllib.request.Request(f"http://{LLAMA_HOST}:{LLAMA_PORT}/health")
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    status_code = getattr(resp, "status", None) or getattr(resp, "code", None)
                    if status_code == 200:
                        cls._is_loading = False
                        cls._last_status_message = "Engine online and ready"
                        return True
            except urllib.error.HTTPError as he:
                # HTTP 503 Service Unavailable is returned while llama-server loads weights/mmproj into memory
                if he.code == 503:
                    cls._is_loading = True
                    cls._last_status_message = "Loading model into VRAM..."
                    time.sleep(1.0)
                    continue
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

        cls._is_loading = False
        cls._last_status_message = "Startup timed out"
        return False



