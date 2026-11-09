"""Relevance evaluation: TF-IDF vs BM25 on test collections with human judgments.

    python benchmarks/evaluate.py            # downloads Cranfield + SciFact if missing

Reports P@10, R@10, MAP (over the top 1000) and nDCG@10 (graded), plus MRR,
for:

* TF-IDF (sqrt tf, ClassicSimilarity-style) and TF-IDF (log tf)
* BM25 with the standard parameters (k1=1.2, b=0.75)
* a k1 x b sensitivity grid. The best cell is tuned *on the test queries*, so
  it is an optimistic upper bound, reported separately for exactly that reason
* analyzer ablations for BM25: no stemming, no stopword removal

Outputs: benchmarks/results/relevance.{md,csv,json} and docs/images/relevance.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import charts  # noqa: E402
from minisearch import AnalyzerConfig, Index  # noqa: E402
from minisearch.evaluation import (  # noqa: E402
    Collection,
    evaluate_run,
    load_beir,
    load_cranfield,
    paired_randomization_test,
    per_query,
)
from minisearch.query.parser import escape  # noqa: E402
from minisearch.ranking import BM25, Ranker, TfIdf  # noqa: E402

RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "benchmarks" / "results"
IMAGES = ROOT / "docs" / "images"
CRAN_URL = "https://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def ensure_datasets() -> tuple[Path, Path]:
    cran_dir = RAW / "cranfield" / "x"
    if not (cran_dir / "cran.all.1400").exists():
        cran_dir.mkdir(parents=True, exist_ok=True)
        tgz = RAW / "cranfield" / "cran.tar.gz"
        print(f"downloading {CRAN_URL}")
        urllib.request.urlretrieve(CRAN_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(cran_dir, filter="data")
    sci_dir = RAW / "scifact"
    if not (sci_dir / "corpus.jsonl").exists():
        z = RAW / "scifact.zip"
        print(f"downloading {SCIFACT_URL}")
        urllib.request.urlretrieve(SCIFACT_URL, z)
        with zipfile.ZipFile(z) as f:
            f.extractall(RAW)
    return cran_dir, sci_dir


def build_index(coll: Collection, cfg: AnalyzerConfig, workdir: Path) -> Index:
    path = workdir / f"{coll.name}-{abs(hash(json.dumps(cfg.to_dict(), sort_keys=True)))}"
    if path.exists():
        shutil.rmtree(path)
    idx = Index(path, analyzer_config=cfg, max_buffered_docs=10**9)
    for d in coll.docs:
        idx.add(d)
    idx.force_merge()
    return idx


def run_queries(
    idx: Index, coll: Collection, ranker: Ranker, k: int = 1000
) -> dict[str, list[str]]:
    s = idx.searcher()
    run = {}
    for qid, text in coll.queries.items():
        res = s.search(escape(text), k=k, ranker=ranker)
        run[qid] = [h.fields["id"] for h in res.hits]
    return run


SIGNIFICANCE: dict[str, dict] = {}


def evaluate(coll: Collection, workdir: Path) -> list[dict]:
    rows: list[dict] = []
    runs: dict[str, dict] = {}
    default_idx = build_index(coll, AnalyzerConfig(), workdir)

    def record(model: str, ranker: Ranker, idx: Index = default_idx, group: str = "main"):
        t0 = time.perf_counter()
        run = run_queries(idx, coll, ranker)
        runs[model] = run
        metrics = evaluate_run(run, coll.qrels).as_dict()
        rows.append(
            {
                "dataset": coll.name,
                "group": group,
                "model": model,
                **metrics,
                "seconds": round(time.perf_counter() - t0, 2),
            }
        )
        print(
            f"  {coll.name:10s} {model:34s} nDCG@10={metrics['nDCG@10']:.4f} MAP={metrics['MAP']:.4f}"
        )

    record("TF-IDF (sqrt tf)", TfIdf("sqrt"))
    record("TF-IDF (log tf)", TfIdf("log"))
    record("BM25 (k1=1.2, b=0.75)", BM25(1.2, 0.75))

    for k1 in (0.4, 0.8, 1.2, 1.6, 2.0):
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            record(f"BM25 k1={k1} b={b}", BM25(k1, b), group="grid")

    for label, cfg in [
        ("BM25, no stemming", AnalyzerConfig(stemmer=None)),
        ("BM25, no stopword removal", AnalyzerConfig(stopwords="none")),
        ("BM25, larger stopword list", AnalyzerConfig(stopwords="english")),
    ]:
        record(label, BM25(), build_index(coll, cfg, workdir), group="ablation")

    # Is BM25 significantly better than TF-IDF? Paired sign-flip randomisation test.
    sig = {}
    bm25 = "BM25 (k1=1.2, b=0.75)"
    for other in ("TF-IDF (sqrt tf)", "TF-IDF (log tf)", "BM25, no stemming"):
        for metric in ("nDCG@10", "AP"):
            a = per_query(runs[bm25], coll.qrels, metric)
            b = per_query(runs[other], coll.qrels, metric)
            delta = sum(a.values()) / len(a) - sum(b.values()) / len(b)
            sig[f"{bm25} vs {other} [{metric}]"] = {
                "delta": round(delta, 4),
                "p": round(paired_randomization_test(a, b), 4),
            }
    SIGNIFICANCE[coll.name] = sig
    return rows


def fmt_table(rows: list[dict], cols=("P@10", "R@10", "MAP", "nDCG@10", "MRR")) -> str:
    out = ["| Model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    best = {c: max(r[c] for r in rows) for c in cols}
    for r in rows:
        cells = [(f"**{r[c]:.4f}**" if r[c] == best[c] else f"{r[c]:.4f}") for c in cols]
        out.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def grid_table(rows: list[dict], metric: str = "nDCG@10") -> str:
    grid = {}
    for r in rows:
        _, k1s, bs = r["model"].split()
        grid[(float(k1s[3:]), float(bs[2:]))] = r[metric]
    k1s = sorted({k for k, _ in grid})
    bs = sorted({b for _, b in grid})
    best = max(grid.values())
    out = ["| k1 \\ b | " + " | ".join(f"{b:g}" for b in bs) + " |", "|---|" + "---:|" * len(bs)]
    for k1 in k1s:
        cells = [
            (f"**{grid[(k1, b)]:.4f}**" if grid[(k1, b)] == best else f"{grid[(k1, b)]:.4f}")
            for b in bs
        ]
        out.append(f"| {k1:g} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_outputs(all_rows: list[dict], collections: list[Collection]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    (RESULTS / "relevance.json").write_text(
        json.dumps({"rows": all_rows, "significance": SIGNIFICANCE}, indent=1)
    )
    with open(RESULTS / "relevance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    md = ["# Relevance evaluation", ""]
    md.append(
        "Free-text queries (syntax characters escaped), ranked OR retrieval, top 1000. "
        "MAP is over the top 1000. nDCG@10 uses graded judgments where available "
        "(Cranfield: gain = 5 - code).\n"
    )
    for coll in collections:
        rows = [r for r in all_rows if r["dataset"] == coll.name]
        n_q = rows[0]["queries"]
        md.append(f"## {coll.name} ({len(coll.docs):,} docs, {n_q} judged queries)\n")
        main = [r for r in rows if r["group"] == "main"]
        grid = [r for r in rows if r["group"] == "grid"]
        best = max(grid, key=lambda r: r["nDCG@10"])
        tuned = dict(best, model=f"BM25 tuned on test ({best['model'][5:]}) †")
        abl = [r for r in rows if r["group"] == "ablation"]
        md.append(fmt_table(main + [tuned] + abl))
        md.append(
            "\n† Parameters chosen on the evaluation queries themselves: an optimistic upper bound, "
            "not a fair comparison.\n"
        )
        md.append("Paired randomisation test (20,000 sign flips, two-sided):\n")
        md.append("| Comparison | Δ (BM25 − other) | p-value |\n|---|---:|---:|")
        for name, v in SIGNIFICANCE[coll.name].items():
            star = " ✱" if v["p"] < 0.05 else ""
            md.append(f"| {name} | {v['delta']:+.4f} | {v['p']:.4f}{star} |")
        md.append("\n✱ significant at α = 0.05\n")
        md.append(f"BM25 nDCG@10 over k1 × b:\n\n{grid_table(grid)}\n")
    (RESULTS / "relevance.md").write_text("\n".join(md))

    fig, ax = charts.setup(8, 4.2)
    groups, series = [], {
        "TF-IDF (sqrt tf)": [],
        "TF-IDF (log tf)": [],
        "BM25 (k1=1.2, b=0.75)": [],
    }
    for coll in collections:
        for metric in ("nDCG@10", "MAP"):
            groups.append(f"{coll.name}\n{metric}")
            for model in series:
                r = next(r for r in all_rows if r["dataset"] == coll.name and r["model"] == model)
                series[model].append(r[metric])
    charts.grouped_bars(ax, groups, series)
    ax.set_title("Relevance: BM25 vs TF-IDF")
    ax.set_ylabel("score (higher is better)")
    ax.set_ylim(0, max(max(v) for v in series.values()) * 1.18)
    charts.save(fig, IMAGES / "relevance.png")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cran_dir, sci_dir = ensure_datasets()
    collections = [load_cranfield(cran_dir), load_beir(sci_dir)]
    all_rows: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        for coll in collections:
            all_rows += evaluate(coll, Path(tmp))
    write_outputs(all_rows, collections)
    print(f"\nwrote {RESULTS / 'relevance.md'} and {IMAGES / 'relevance.png'}")


if __name__ == "__main__":
    main()
