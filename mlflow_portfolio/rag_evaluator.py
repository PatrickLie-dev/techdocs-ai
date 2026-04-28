"""
MLflow RAG Evaluation Pipeline — TechDocs AI
Tracks retrieval quality, response metrics, and prompt experiments

API Contract (techdocs-ai-production.up.railway.app):
  POST /api/chat   { "question": "..." }
  -> { "answer": "...", "sources": [{ "source": "file.pdf", "page": 3 }] }
  GET  /api/health -> { "status": "ok", "indexed_chunks": N }
"""

import mlflow
import time
import os
import statistics
import json
import requests
from dataclasses import dataclass, field
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "./mlruns"
)
EXPERIMENT_NAME     = "TechDocs-RAG-Evaluation"
TECHDOCS_URL        = os.getenv(
    "TECHDOCS_URL",
    "https://techdocs-ai-production.up.railway.app"
)

# ── Evaluation queries (domain: tech docs / Linux / RAG) ──────────────────────
EVAL_QUERIES = [
    {
        "id": "q1",
        "query": "What is Linux and how does it work?",
        "category": "concept",
        "expected_keywords": ["linux", "kernel", "operating system", "open source"],
    },
    {
        "id": "q2",
        "query": "How do I list files in a directory using the terminal?",
        "category": "command",
        "expected_keywords": ["ls", "directory", "terminal", "command"],
    },
    {
        "id": "q3",
        "query": "What is the difference between RAM and ROM?",
        "category": "concept",
        "expected_keywords": ["ram", "rom", "memory", "storage", "volatile"],
    },
    {
        "id": "q4",
        "query": "How does a RAG pipeline retrieve documents?",
        "category": "rag",
        "expected_keywords": ["retrieval", "embedding", "vector", "similarity", "chunk"],
    },
    {
        "id": "q5",
        "query": "What is a firewall and why is it important?",
        "category": "networking",
        "expected_keywords": ["firewall", "network", "security", "traffic", "block"],
    },
]


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    query_id:        str
    query:           str
    category:        str
    answer:          str
    latency_ms:      float
    num_sources:     int
    source_files:    list
    answer_length:   int
    relevance_score: float
    status:          str = "success"
    error:           Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def keyword_relevance(answer: str, keywords: list[str]) -> float:
    """Fraction of expected keywords found in the answer (0.0 – 1.0)."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return round(hits / len(keywords), 3) if keywords else 0.0


def check_health() -> dict:
    """Hit /api/health before running evals."""
    try:
        r = requests.get(f"{TECHDOCS_URL}/api/health", timeout=10)
        return r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def query_techdocs(question: str) -> dict:
    """POST /api/chat and return structured result."""
    start = time.time()
    try:
        response = requests.post(
            f"{TECHDOCS_URL}/api/chat",
            json={"question": question},
            timeout=30,
        )
        latency_ms = (time.time() - start) * 1000

        if response.status_code == 200:
            data = response.json()
            sources = data.get("sources", [])
            return {
                "answer":      data.get("answer", ""),
                "sources":     sources,
                "num_sources": len(sources),
                "latency_ms":  latency_ms,
                "status":      "success",
            }
        else:
            return {
                "answer": "", "sources": [], "num_sources": 0,
                "latency_ms": latency_ms,
                "status": f"http_{response.status_code}",
                "error":  response.text[:200],
            }

    except requests.exceptions.Timeout:
        return {"answer": "", "sources": [], "num_sources": 0,
                "latency_ms": (time.time() - start) * 1000,
                "status": "timeout", "error": "Request timed out after 30s"}
    except Exception as e:
        return {"answer": "", "sources": [], "num_sources": 0,
                "latency_ms": (time.time() - start) * 1000,
                "status": "error", "error": str(e)}


# ── Main experiment runner ────────────────────────────────────────────────────
def run_rag_experiment(
    run_name:    str,
    description: str = "",
) -> dict:
    """
    Evaluate TechDocs AI against EVAL_QUERIES and log everything to MLflow.

    Args:
        run_name:    Descriptive name shown in MLflow UI
        description: Notes about what changed vs previous run
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Pre-flight health check ────────────────────────────────────────────
    print(f"\n🔍 Checking TechDocs AI health at {TECHDOCS_URL} ...")
    health = check_health()
    print(f"   Health: {health}")

    if health.get("status") != "ok":
        print("⚠️  Warning: health check failed. Proceeding anyway...")

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n🚀 MLflow run started: {run.info.run_id}")

        # ── Log run config ─────────────────────────────────────────────────
        mlflow.log_params({
            "techdocs_url":      TECHDOCS_URL,
            "eval_query_count":  len(EVAL_QUERIES),
            "indexed_chunks":    health.get("indexed_chunks", "unknown"),
            "eval_method":       "keyword_heuristic",
            "description":       description or "baseline",
        })

        mlflow.set_tags({
            "project":    "TechDocs-AI",
            "deployment": "Railway",
            "url":        "techdocs-ai-production.up.railway.app",
            "owner":      "Patrick Lie",
            "month":      "Month-4-MLOps",
        })

        # ── Run all queries ────────────────────────────────────────────────
        results: list[QueryResult] = []

        for i, item in enumerate(EVAL_QUERIES, 1):
            print(f"\n  [{i}/{len(EVAL_QUERIES)}] ({item['category']}) {item['query']}")

            api = query_techdocs(item["query"])

            if api["status"] == "success":
                relevance = keyword_relevance(api["answer"], item["expected_keywords"])
                result = QueryResult(
                    query_id=        item["id"],
                    query=           item["query"],
                    category=        item["category"],
                    answer=          api["answer"],
                    latency_ms=      api["latency_ms"],
                    num_sources=     api["num_sources"],
                    source_files=    [s.get("source","") for s in api["sources"]],
                    answer_length=   len(api["answer"].split()),
                    relevance_score= relevance,
                    status=          "success",
                )
                print(f"     ✅ {result.latency_ms:.0f}ms | "
                      f"sources={result.num_sources} | "
                      f"relevance={result.relevance_score:.2f} | "
                      f"words={result.answer_length}")
            else:
                result = QueryResult(
                    query_id=item["id"], query=item["query"],
                    category=item["category"], answer="",
                    latency_ms=api["latency_ms"], num_sources=0,
                    source_files=[], answer_length=0,
                    relevance_score=0.0,
                    status=api["status"], error=api.get("error"),
                )
                print(f"     ❌ Failed: {api['status']} — {api.get('error','')[:80]}")

            results.append(result)

            # Per-query metrics (step = query index → creates time-series chart)
            mlflow.log_metric("latency_ms",      result.latency_ms,      step=i)
            mlflow.log_metric("relevance_score",  result.relevance_score, step=i)
            mlflow.log_metric("num_sources",      result.num_sources,     step=i)
            mlflow.log_metric("answer_length",    result.answer_length,   step=i)

        # ── Aggregate metrics ──────────────────────────────────────────────
        successful = [r for r in results if r.status == "success"]
        failed     = [r for r in results if r.status != "success"]

        if successful:
            latencies  = [r.latency_ms      for r in successful]
            relevances = [r.relevance_score for r in successful]
            sources    = [r.num_sources     for r in successful]
            lengths    = [r.answer_length   for r in successful]

            agg = {
                "success_rate":       round(len(successful) / len(results), 3),
                "avg_latency_ms":     round(statistics.mean(latencies),     1),
                "p95_latency_ms":     round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
                "max_latency_ms":     round(max(latencies),  1),
                "avg_relevance":      round(statistics.mean(relevances),    3),
                "min_relevance":      round(min(relevances),  3),
                "avg_sources":        round(statistics.mean(sources),       2),
                "avg_answer_words":   round(statistics.mean(lengths),       1),
                "failed_queries":     len(failed),
            }
        else:
            agg = {"success_rate": 0.0, "failed_queries": len(failed)}

        mlflow.log_metrics(agg)

        print(f"\n  📊 Summary:")
        for k, v in agg.items():
            print(f"     {k}: {v}")

        # ── Save eval report as artifact ───────────────────────────────────
        report = _build_report(run_name, results, agg, health)
        with open("eval_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact("eval_report.md")
        os.remove("eval_report.md")

        # Save raw results as JSON artifact
        raw = [
            {
                "id": r.query_id, "query": r.query, "category": r.category,
                "status": r.status, "latency_ms": round(r.latency_ms, 1),
                "relevance_score": r.relevance_score, "num_sources": r.num_sources,
                "source_files": r.source_files, "answer_words": r.answer_length,
                "answer_preview": r.answer[:200] + "..." if len(r.answer) > 200 else r.answer,
            }
            for r in results
        ]
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
        mlflow.log_artifact("eval_results.json")
        os.remove("eval_results.json")

        print(f"\n✨ Done! Run ID: {run.info.run_id}")
        print(f"   mlflow ui --port 5001  →  http://localhost:5001")
        return {"run_id": run.info.run_id, "metrics": agg}


# ── Report builder ────────────────────────────────────────────────────────────
def _build_report(run_name: str, results: list, agg: dict, health: dict) -> str:
    lines = [
        "# TechDocs AI — RAG Evaluation Report",
        f"**Run:** {run_name}",
        f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Service:** {TECHDOCS_URL}",
        f"**Indexed chunks:** {health.get('indexed_chunks', 'unknown')}",
        "",
        "## Summary Metrics",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in agg.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Per-Query Results",
        "| # | Category | Query | Latency | Relevance | Sources | Words | Status |",
        "|---|----------|-------|---------|-----------|---------|-------|--------|",
    ]
    for i, r in enumerate(results, 1):
        query_short = r.query[:45] + "..." if len(r.query) > 45 else r.query
        status_icon = "✅" if r.status == "success" else "❌"
        lines.append(
            f"| {i} | {r.category} | {query_short} | "
            f"{r.latency_ms:.0f}ms | {r.relevance_score:.2f} | "
            f"{r.num_sources} | {r.answer_length} | {status_icon} {r.status} |"
        )

    lines += ["", "## Source Coverage"]
    all_sources = set()
    for r in results:
        all_sources.update(r.source_files)
    for s in sorted(all_sources):
        if s:
            lines.append(f"- {s}")

    lines += ["", "---", "*Generated by MLflow RAG Evaluation Pipeline — Patrick Lie Portfolio*"]
    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Baseline run
    run_rag_experiment(
        run_name    = "techdocs-baseline-eval",
        description = "Initial baseline evaluation of deployed TechDocs AI on Railway",
    )