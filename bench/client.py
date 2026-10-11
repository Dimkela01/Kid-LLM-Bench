"""Streaming chat request to llama-server with client-side timing."""
import json
import re
import time

import httpx

THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.S)
SENTENCE_END = re.compile(r"[.!?]\s")


def strip_think(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> blocks (and an unfinished trailing one)."""
    if "<think>" not in text:
        return text, False
    return THINK_BLOCK.sub("", text).strip(), True


def chat(base_url: str, messages: list, params: dict, timeout_s: float) -> dict:
    """Send one streaming chat request. Raises on HTTP error or timeout."""
    body = {"messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, **params}
    raw, reasoning = "", ""
    ttft = ttfs = punct_ms = punct_pos = None
    finish_reason, timings, usage = None, None, None

    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    with httpx.stream("POST", base_url + "/v1/chat/completions", json=body,
                      timeout=httpx.Timeout(timeout_s)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if time.perf_counter() > deadline:
                raise TimeoutError(f"request exceeded {timeout_s}s")
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings") or timings
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                reasoning += delta.get("reasoning_content") or ""
                raw += delta.get("content") or ""
                finish_reason = choice.get("finish_reason") or finish_reason
            visible, _ = strip_think(raw)
            now_ms = (time.perf_counter() - t0) * 1000
            if ttft is None and visible.strip():
                ttft = now_ms
            if ttfs is None:
                # The first sentence ends at a . ! or ? followed by whitespace. The
                # punctuation usually arrives one token before the space, so remember
                # when it arrived and use that time once the space confirms it.
                m = SENTENCE_END.search(visible)
                if m:
                    ttfs = punct_ms if punct_pos == m.start() else now_ms
                elif visible and visible[-1] in ".!?":
                    punct_pos, punct_ms = len(visible) - 1, now_ms
    total = (time.perf_counter() - t0) * 1000

    visible, had_think = strip_think(raw)
    answer = visible.strip()
    if ttfs is None:
        # One sentence ending at the end of the text, or no punctuation at all (= total).
        ends_here = punct_pos is not None and punct_pos == len(visible.rstrip()) - 1
        ttfs = punct_ms if ends_here else total
    return {
        "answer": answer,
        "had_think": int(had_think or bool(reasoning.strip())),
        "truncated": int(finish_reason == "length"),
        "ttft_ms": ttft if ttft is not None else total,
        "ttfs_ms": ttfs,
        "total_ms": total,
        "timings": timings,
        "usage": usage,
    }


def tokenize_count(base_url: str, text: str) -> int:
    """Fallback token count via llama-server's /tokenize endpoint."""
    r = httpx.post(base_url + "/tokenize", json={"content": text}, timeout=30)
    r.raise_for_status()
    return len(r.json()["tokens"])
