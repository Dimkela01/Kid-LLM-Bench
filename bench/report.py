"""Write summary.md, rating_sheet.csv and rating_key.csv for one results directory.

Usage: python -m bench.report results/<YYYY-MM-DD_HHMM>
"""
import json
import random
import string
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RATING_COLS = ["correct", "age_appropriate", "safe", "unnecessary_refusal", "notes"]


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def pct(s: pd.Series) -> str:
    return f"{100 * s.mean():.0f}% (n={len(s)})" if len(s) else "n/a"


def ms(s: pd.Series, q: float) -> str:
    s = num(s).dropna()
    return f"{s.quantile(q):.0f}" if len(s) else "n/a"


def cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def md_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def model_summary(r: pd.DataFrame, m: pd.Series) -> str:
    ok = r[r["error"] == ""]
    sr = ok[ok["lang"] == "sr"]
    en = ok[ok["lang"] == "en"]
    rows = [
        ["TTFT ms (median / p95)", f"{ms(ok.ttft_ms, .5)} / {ms(ok.ttft_ms, .95)}"],
        ["TTFS ms (median / p95)", f"{ms(ok.ttfs_ms, .5)} / {ms(ok.ttfs_ms, .95)}"],
        ["Total ms (median / p95)", f"{ms(ok.total_ms, .5)} / {ms(ok.total_ms, .95)}"],
        ["Prefill tok/s (median)", ms(ok.prefill_tps, .5)],
        ["Decode tok/s (median)", ms(ok.decode_tps, .5)],
        ["Peak RSS MB (working set)", m.peak_rss_mb],
        ["Peak private MB (committed)", m.peak_private_mb],
        ["Idle RSS / private MB", f"{m.idle_rss_mb} / {m.idle_private_mb}"],
        ["File size MB", m.file_size_mb],
        ["Load time s", m.load_time_s],
        ["Cold TTFT ms (warm-up)", m.cold_ttft_ms],
        ["Uncached TTFT ms (median)", m.uncached_ttft_ms_median],
        ["Avg words per answer", f"{num(ok.words).mean():.1f}" if len(ok) else "n/a"],
        ["Answers with ≤3 sentences", pct(num(ok.sentences) <= 3)],
        ["lang_ok — all", pct(num(ok.lang_ok) == 1)],
        ["lang_ok — English questions", pct(num(en.lang_ok) == 1)],
        ["lang_ok — Serbian questions", pct(num(sr.lang_ok) == 1)],
        ["tricky_unsafe mentioning a grown-up", pct(num(ok[ok.group == "tricky_unsafe"].mentions_grownup) == 1)],
        ["tricky_benign mentioning a grown-up", pct(num(ok[ok.group == "tricky_benign"].mentions_grownup) == 1)],
        ["Think tags / truncations / errors",
         f"{int(num(ok.had_think).sum())} / {int(num(ok.truncated).sum())} / {int((r.error != '').sum())}"],
    ]
    return md_table(["metric", "value"], rows)


def main(out: Path) -> None:
    runs = pd.read_csv(out / "runs.csv", dtype=str, keep_default_na=False)
    models = pd.read_csv(out / "models.csv", dtype=str, keep_default_na=False)
    env = json.loads((out / "env.json").read_text(encoding="utf-8"))
    qfile = ROOT / env.get("questions_file", "data/questions_v1.jsonl")
    questions = {q["id"]: q for q in map(json.loads, qfile.read_text(encoding="utf-8").splitlines()) if q}

    # ---- summary.md
    parts = [f"# Benchmark summary — {out.name}", "",
             f"- CPU: {env['cpu']} ({env['physical_cores']} physical cores), RAM {env['total_ram_mb']} MB "
             f"({env.get('available_ram_mb_at_start', 'n/a')} MB free at start)",
             f"- OS: {env['os']}", f"- llama.cpp: {env['llama_cpp_version']}",
             f"- Runtime: {env.get('start_time')} → {env.get('end_time')} ({env.get('total_runtime_s', 'n/a')} s)",
             f"- Runs: {len(runs)} rows in runs.csv", ""]
    for _, m in models.iterrows():
        parts += [f"## {m.model_key}", "", f"`{m.repo}/{m.filename}`", "",
                  model_summary(runs[runs.model_key == m.model_key], m), ""]

    first = runs[runs["repeat"] == "1"]
    ids = sorted(first.question_id.unique())
    sample = random.Random(0).sample(ids, min(5, len(ids)))
    parts += ["## 5 random questions, side by side", ""]
    for qid in sample:
        q = questions[qid]
        parts += [f"### {qid} ({q['lang']}, {q['group']}, expected: {q['expected']})", "",
                  f"> {q['question']}", "", f"_Note: {q['note']}_", ""]
        rows = [[r.model_key, r.answer if r.error == "" else f"ERROR: {r.error}"]
                for r in first[first.question_id == qid].itertuples()]
        parts += [md_table(["model", "answer"], rows), ""]
    (out / "summary.md").write_text("\n".join(parts), encoding="utf-8")

    # ---- blind rating sheet + key (unseeded RNG so the code mapping can't be predicted)
    rng = random.Random()
    keys = list(models.model_key)
    rng.shuffle(keys)
    code_of = {k: string.ascii_uppercase[i] for i, k in enumerate(keys)}
    pd.DataFrame({"code": list(code_of.values()), "model_key": list(code_of.keys())}).sort_values("code") \
        .to_csv(out / "rating_key.csv", index=False, encoding="utf-8-sig")

    rated = runs[runs["error"] == ""].sample(frac=1, random_state=rng.randrange(2**32)).reset_index(drop=True)
    sheet = pd.DataFrame({
        "row_id": range(1, len(rated) + 1),
        "code": rated.model_key.map(code_of),
        "question_id": rated.question_id,
        "group": rated.group,
        "expected": rated.expected,
        "question": rated.question_id.map(lambda i: questions[i]["question"]),
        "answer": rated.answer,
        **{c: "" for c in RATING_COLS},
    })
    sheet.to_csv(out / "rating_sheet.csv", index=False, encoding="utf-8-sig")
    n_err = int((runs.error != "").sum())
    print(f"Wrote summary.md, rating_sheet.csv ({len(sheet)} rows"
          f"{f', {n_err} errored runs left out' if n_err else ''}), rating_key.csv in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
