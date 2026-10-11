"""Start/stop one llama-server process, wait for /health, sample its RSS."""
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx
import psutil


def find_llama_server(configured: str | None) -> str:
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("llama-server")
    if not found:
        raise FileNotFoundError("llama-server not found; set paths.llama_server in config.yaml")
    return found


def llama_server_version(binary: str) -> str:
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    return " ".join((out.stdout + out.stderr).split())


def private_bytes(mi) -> int | None:
    """Committed private memory (psutil reports it on Windows only)."""
    return getattr(mi, "private", None)


class RssSampler:
    """Background thread that tracks peak RSS (and private bytes, on Windows) since reset().

    On Windows psutil's RSS is the working set, which the OS can trim far below what
    the process actually allocated, so committed private bytes are tracked as well.
    """

    def __init__(self, pid: int, interval_s: float):
        self.proc = psutil.Process(pid)
        self.interval_s = interval_s
        self.peak = 0
        self.peak_private = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                mi = self.proc.memory_info()
                self.peak = max(self.peak, mi.rss)
                self.peak_private = max(self.peak_private, private_bytes(mi) or 0)
            except psutil.Error:
                return
            time.sleep(self.interval_s)

    def current_mb(self) -> tuple[float, float | None]:
        """(rss_mb, private_mb); private is None where the OS doesn't report it."""
        mi = self.proc.memory_info()
        priv = private_bytes(mi)
        return mi.rss / 2**20, (priv / 2**20 if priv is not None else None)

    def reset(self) -> None:
        mi = self.proc.memory_info()
        self.peak, self.peak_private = mi.rss, private_bytes(mi) or 0

    def peak_mb(self) -> float:
        return self.peak / 2**20

    def peak_private_mb(self) -> float | None:
        return self.peak_private / 2**20 if self.peak_private else None

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


class LlamaServer:
    def __init__(self, binary: str, model_path: Path, cfg: dict, extra_args: list, log_path: Path):
        s = cfg["server"]
        self.host, self.port = s["host"], s["port"]
        self.base_url = f"http://{s['host']}:{s['port']}"
        self.cmd = [
            binary, "-m", str(model_path),
            "-c", str(s["n_ctx"]), "-t", str(s["threads"]), "-ngl", str(s["ngl"]),
            "--host", s["host"], "--port", str(s["port"]),
            *s["common_args"], *extra_args,
        ]
        self.health_timeout_s = s["health_timeout_s"]
        self.rss_interval_s = cfg["request"]["rss_sample_ms"] / 1000
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.sampler: RssSampler | None = None

    def start(self) -> float:
        """Start the server and block until /health is OK. Returns load time in seconds."""
        with socket.socket() as sock:
            if sock.connect_ex((self.host, self.port)) == 0:
                raise RuntimeError(f"port {self.port} is already in use; change server.port in config.yaml")
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._log.write(" ".join(self.cmd) + "\n\n")
        self._log.flush()
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self.sampler = RssSampler(self.proc.pid, self.rss_interval_s)
        while time.perf_counter() - t0 < self.health_timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
            try:
                if httpx.get(self.base_url + "/health", timeout=2).status_code == 200:
                    return time.perf_counter() - t0
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"llama-server not healthy after {self.health_timeout_s}s")

    def stop(self) -> None:
        if self.sampler:
            self.sampler.stop()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "_log", None):
            self._log.close()
