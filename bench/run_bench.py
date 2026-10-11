"""Run every model x question once (tricky_unsafe 3x) and write runs.csv, models.csv, env.json.

Usage (from the project root):
    python -m bench.run_bench                 # full run, all models
    python -m bench.run_bench --dry-run       # first model, 5 questions
    python -m bench.run_bench --models gemma3-1b --questions sci-001,dan-002
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"  # the benchmark never talks to the Hub

import argparse
import csv
import json
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml
from lingua import Language, LanguageDetectorBuilder

from bench.client import chat, tokenize_count
from bench.server import LlamaServer, find_llama_server, llama_server_version

ROOT = Path(__file__).resolve().parent.parent

RUN_FIELDS = [
    "model_key", "quant", "question_id", "lang", "group", "category", "expected", "repeat", "seed",
    "answer", "had_think", "truncated", "error",
    "ttft_ms", "ttfs_ms", "total_ms",
    "prompt_tokens", "prompt_tokens_processed", "output_tokens", "prefill_tps", "decode_tps",
    "metrics_source", "peak_rss_mb", "words", "sentences", "lang_ok", "mentions_grownup",
]
MODEL_FIELDS = [
    "model_key", "repo", "filename", "file_size_mb", "load_time_s", "idle_rss_mb", "peak_rss_mb",
    "idle_private_mb", "peak_private_mb", "cold_ttft_ms", "uncached_ttft_ms_median", "llama_cpp_version", "threads", "n_ctx",
]


# ---------------------------------------------------------------- environment

def cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "n/a"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a (not a git repo)"


def environment(llama_version: str) -> dict:
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_mb": round(psutil.virtual_memory().total / 2**20),
        "available_ram_mb_at_start": round(psutil.virtual_memory().available / 2**20),
        "llama_cpp_version": llama_version,
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
    }


def fmt_mb(mb: float | None):
    return round(mb) if mb is not None else "n/a"


# ---------------------------------------------------------------- text checks

class TextChecks:
    def __init__(self, keywords: dict):
        words = "|".join(re.escape(w) for w in keywords["words"])
        stems = "|".join(re.escape(s) for s in keywords["stems"])
        self.grownup = re.compile(rf"\b(?:{words})\b|\b(?:{stems})\w*", re.I)
        self.detector = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.SERBIAN, Language.CROATIAN, Language.BOSNIAN).build()

    @staticmethod
    def words(text: str) -> int:
        return len(re.findall(r"\w+(?:['’-]\w+)*", text))

    @staticmethod
    def sentences(text: str) -> int:
        return len([s for s in re.split(r"[.!?]+(?:\s+|$)", text) if s.strip()])

    def lang_ok(self, text: str, lang: str) -> int:
        detected = self.detector.detect_language_of(text) if text.strip() else None
        if detected is None:
            return 0
        got = "en" if detected == Language.ENGLISH else "sr"  # hr/bs count as Serbian (Latin)
        return int(got == lang)

    def mentions_grownup(self, text: str) -> int:
        return int(bool(self.grownup.search(text)))


# ---------------------------------------------------------------- requests

def request(server: LlamaServer, messages: list, params: dict, req_cfg: dict) -> tuple[dict | None, float, str]:
    """One request with retries. Returns (result, peak_rss_mb, error)."""
    error = ""
    for attempt in range(1 + req_cfg["retries"]):
        server.sampler.reset()
        try:
            result = chat(server.base_url, messages, params, req_cfg["timeout_s"])
            return result, server.sampler.peak_mb(), ""
        except Exception as e:  # timeout, HTTP error, server crash...
            error = f"{type(e).__name__}: {e}"[:300]
            print(f"    attempt {attempt + 1} failed: {error}")
    return None, server.sampler.peak_mb(), error


def token_metrics(server: LlamaServer, result: dict, messages: list) -> dict:
    t, u = result["timings"], result["usage"]
    if t and u:
        return {
            "prompt_tokens": t.get("cache_n", 0) + t["prompt_n"],
            "prompt_tokens_processed": t["prompt_n"],
            "output_tokens": u["completion_tokens"],
            "prefill_tps": round(t["prompt_per_second"], 1),
            "decode_tps": round(t["predicted_per_second"], 1),
            "metrics_source": "server",
        }
    # Fallback: count tokens ourselves; prefill speed and cache split are not observable.
    import httpx
    prompt = httpx.post(server.base_url + "/apply-template", json={"messages": messages}, timeout=30).json()["prompt"]
    out = tokenize_count(server.base_url, result["answer"])
    gen_s = (result["total_ms"] - result["ttft_ms"]) / 1000
    return {
        "prompt_tokens": tokenize_count(server.base_url, prompt),
        "prompt_tokens_processed": "n/a",
        "output_tokens": out,
        "prefill_tps": "n/a",
        "decode_tps": round((out - 1) / gen_s, 1) if gen_s > 0 and out > 1 else "n/a",
        "metrics_source": "tokenize",
    }


# ---------------------------------------------------------------- main

def pick_dry_run(questions: list) -> list:
    """5 questions: normal en + sr, tricky_unsafe en + sr, tricky_benign."""
    picks = []
    for group, lang in [("normal", "en"), ("normal", "sr"), ("tricky_unsafe", "en"),
                        ("tricky_unsafe", "sr"), ("tricky_benign", "en")]:
        picks.append(next(q for q in questions if q["group"] == group and q["lang"] == lang))
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated model keys (default: all)")
    ap.add_argument("--questions", help="comma-separated question ids (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="first model x 5 questions")
    ap.add_argument("--out", help="output directory (default: results/<timestamp>)")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    system_prompt = (ROOT / paths["system_prompt"]).read_text(encoding="utf-8").strip()
    all_questions = [json.loads(l) for l in (ROOT / paths["questions"]).read_text(encoding="utf-8").splitlines() if l.strip()]

    models = cfg["models"]
    if args.models:
        keys = args.models.split(",")
        models = [m for m in models if m["key"] in keys]
    questions = all_questions
    if args.questions:
        ids = args.questions.split(",")
        questions = [q for q in all_questions if q["id"] in ids]
    if args.dry_run:
        models, questions = models[:1], pick_dry_run(all_questions)

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out = Path(args.out) if args.out else ROOT / paths["results_dir"] / (stamp + ("_dryrun" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=True)

    binary = find_llama_server(paths.get("llama_server"))
    version = llama_server_version(binary)
    env = environment(version)
    env.update({"start_time": datetime.now().isoformat(timespec="seconds"),
                "models": [m["key"] for m in models], "n_questions": len(questions),
                "questions_file": paths["questions"], "system_prompt_file": paths["system_prompt"]})
    (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(f"Output: {out}\nllama.cpp: {version}\n")

    checks = TextChecks(cfg["grownup_keywords"])
    samp, req_cfg, s = cfg["sampling"], cfg["request"], cfg["server"]
    base_params = {k: samp[k] for k in ("temperature", "top_p", "max_tokens")}
    n_rep = cfg["runs"]["repeats_tricky_unsafe"]
    t_start = time.perf_counter()

    runs_f = open(out / "runs.csv", "w", newline="", encoding="utf-8")
    models_f = open(out / "models.csv", "w", newline="", encoding="utf-8")
    runs_w = csv.DictWriter(runs_f, RUN_FIELDS)
    models_w = csv.DictWriter(models_f, MODEL_FIELDS)
    runs_w.writeheader()
    models_w.writeheader()

    try:
        for m in models:
            model_path = ROOT / paths["models_dir"] / m["filename"]
            print(f"=== {m['key']} ({model_path.stat().st_size / 2**20:.0f} MB)")
            server = LlamaServer(binary, model_path, cfg, m.get("server_args", []), out / f"server_{m['key']}.log")
            try:
                load_s = server.start()
                time.sleep(0.5)
                idle_mb, idle_priv = server.sampler.current_mb()
                print(f"  loaded in {load_s:.1f}s, idle RSS {idle_mb:.0f} MB, private {fmt_mb(idle_priv)} MB")
                extra = m.get("request_extra") or {}

                def msgs(text):
                    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]

                warm, _, err = request(server, msgs(cfg["runs"]["warmup_question"]),
                                       {**base_params, "seed": samp["seed"], **extra}, req_cfg)
                cold_ttft = round(warm["ttft_ms"]) if warm else "n/a"
                print(f"  warm-up TTFT {cold_ttft} ms {err}")

                peak_mb, peak_priv = 0.0, None
                n_total = sum(n_rep if q["group"] == "tricky_unsafe" else 1 for q in questions)
                i = 0
                for q in questions:
                    for rep in range(1, (n_rep if q["group"] == "tricky_unsafe" else 1) + 1):
                        i += 1
                        seed = samp["seed"] + (rep - 1 if samp["vary_seed_per_repeat"] else 0)
                        messages = msgs(q["question"])
                        params = {**base_params, "seed": seed, "cache_prompt": True, **extra}
                        res, rss, err = request(server, messages, params, req_cfg)
                        peak_mb = max(peak_mb, rss)
                        if server.sampler.peak_private_mb() is not None:
                            peak_priv = max(peak_priv or 0, server.sampler.peak_private_mb())
                        row = {"model_key": m["key"], "quant": m["quant"], "question_id": q["id"],
                               "lang": q["lang"], "group": q["group"], "category": q["category"],
                               "expected": q["expected"], "repeat": rep, "seed": seed,
                               "error": err, "peak_rss_mb": round(rss)}
                        if res:
                            a = res["answer"]
                            row.update({
                                "answer": a, "had_think": res["had_think"], "truncated": res["truncated"],
                                "ttft_ms": round(res["ttft_ms"]), "ttfs_ms": round(res["ttfs_ms"]),
                                "total_ms": round(res["total_ms"]),
                                **token_metrics(server, res, messages),
                                "words": checks.words(a), "sentences": checks.sentences(a),
                                "lang_ok": checks.lang_ok(a, q["lang"]),
                                "mentions_grownup": checks.mentions_grownup(a),
                            })
                            if res["had_think"]:
                                print("    !! think block found and stripped")
                        runs_w.writerow(row)
                        runs_f.flush()
                        print(f"  [{i}/{n_total}] {q['id']} r{rep}: ttfs {row.get('ttfs_ms', '-')} ms, "
                              f"{row.get('output_tokens', '-')} tok {'TRUNC ' if row.get('truncated') else ''}"
                              f"{'ERR ' + err if err else ''}")

                # Prompt-cache test: same questions, prompt cache disabled (not rated).
                cache_qs = [q for q in all_questions if q["group"] == "normal"][:cfg["runs"]["cache_test_questions"]]
                ttfts = []
                for q in cache_qs:
                    res, rss, _ = request(server, msgs(q["question"]),
                                          {**base_params, "seed": samp["seed"], "cache_prompt": False, **extra}, req_cfg)
                    if res:
                        ttfts.append(res["ttft_ms"])
                uncached = round(statistics.median(ttfts)) if ttfts else "n/a"
                print(f"  uncached TTFT median {uncached} ms")

                models_w.writerow({
                    "model_key": m["key"], "repo": m["repo"], "filename": m["filename"],
                    "file_size_mb": round(model_path.stat().st_size / 2**20),
                    "load_time_s": round(load_s, 2), "idle_rss_mb": round(idle_mb),
                    "peak_rss_mb": round(peak_mb), "idle_private_mb": fmt_mb(idle_priv),
                    "peak_private_mb": fmt_mb(peak_priv), "cold_ttft_ms": cold_ttft,
                    "uncached_ttft_ms_median": uncached, "llama_cpp_version": version,
                    "threads": s["threads"], "n_ctx": s["n_ctx"],
                })
                models_f.flush()
            finally:
                server.stop()  # also runs on Ctrl-C
    finally:
        runs_f.close()
        models_f.close()
        env["end_time"] = datetime.now().isoformat(timespec="seconds")
        env["total_runtime_s"] = round(time.perf_counter() - t_start)
        (out / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        print(f"\nDone in {env['total_runtime_s'] / 60:.1f} min -> {out}")


if __name__ == "__main__":
    sys.exit(main())
