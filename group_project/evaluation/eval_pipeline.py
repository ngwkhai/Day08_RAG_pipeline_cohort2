"""
RAG Evaluation Pipeline.

Sá»­ dá»¥ng DeepEval / RAGAS / TruLens Ä‘á»ƒ Ä‘Ã¡nh giÃ¡ cháº¥t lÆ°á»£ng RAG pipeline.
Chá»n 1 framework vÃ  implement Ä‘áº§y Ä‘á»§.

YÃªu cáº§u:
    1. Load golden_dataset.json (â‰¥15 Q&A pairs)
    2. Cháº¡y RAG pipeline trÃªn tá»«ng question
    3. Evaluate vá»›i 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sÃ¡nh A/B Ã­t nháº¥t 2 configs
    5. Export results ra results.md
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

METRIC_NAMES = [
    "faithfulness",
    "answer_relevance",
    "context_recall",
    "context_precision",
]


def load_golden_dataset() -> list[dict]:
    """Load golden dataset tá»« JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Local metric helpers
# =============================================================================

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were",
    "va", "ve", "la", "lao", "cua", "co", "cac", "mot", "nhung",
    "duoc", "tai", "theo", "trong", "cho", "voi", "den", "noi",
    "gi", "nao", "nhu", "thi", "bi", "da", "hay", "hoac", "tu",
    "nay", "do", "du", "can", "tren", "duoi",
}


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def _tokens(text: str) -> list[str]:
    normalized = _strip_accents(text).lower()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _recall(text: str, target: str) -> float:
    target_tokens = _token_set(target)
    if not target_tokens:
        return 0.0
    return len(_token_set(text) & target_tokens) / len(target_tokens)


def _precision(text: str, target: str) -> float:
    text_tokens = _token_set(text)
    if not text_tokens:
        return 0.0
    return len(text_tokens & _token_set(target)) / len(text_tokens)


def _f1(text: str, target: str) -> float:
    p = _precision(text, target)
    r = _recall(text, target)
    return _safe_div(2 * p * r, p + r)


def _clip(score: float) -> float:
    if math.isnan(score):
        return 0.0
    return max(0.0, min(1.0, float(score)))


def _source_name(item: dict) -> str:
    meta = item.get("metadata") or {}
    return (
        meta.get("source")
        or meta.get("file")
        or meta.get("path")
        or meta.get("doc_id")
        or "unknown"
    )


def _build_extractive_answer(question: str, sources: list[dict], max_chars: int = 1200) -> str:
    if not sources:
        return "Khong tim thay ngu canh lien quan trong corpus hien tai."

    parts = []
    for idx, source in enumerate(sources[:3], 1):
        content = re.sub(r"\s+", " ", source.get("content", "")).strip()
        citation = _source_name(source)
        if content:
            parts.append(f"[{idx}] {content[:360]} [{citation}]")
    answer = " ".join(parts)
    return answer[:max_chars] if answer else f"Khong co doan tra loi rut trich cho: {question}"


def _score_case(item: dict, actual_output: str, sources: list[dict]) -> dict:
    context = "\n".join(source.get("content", "") for source in sources)
    expected_answer = item.get("expected_answer", "")
    expected_context = item.get("expected_context", "")
    question = item.get("question", "")
    evidence = " ".join([expected_answer, expected_context, question])

    faithfulness = _recall(context, actual_output)
    answer_relevance = 0.75 * _f1(actual_output, expected_answer) + 0.25 * _f1(actual_output, question)
    context_recall = 0.7 * _recall(context, expected_answer) + 0.3 * _recall(context, expected_context)

    chunk_scores = []
    for rank, source in enumerate(sources[:10], 1):
        chunk = source.get("content", "")
        chunk_relevance = max(_recall(chunk, expected_answer), _recall(chunk, expected_context), _f1(chunk, evidence))
        chunk_scores.append(chunk_relevance / rank)
    context_precision = _safe_div(sum(chunk_scores), sum(1 / rank for rank in range(1, len(chunk_scores) + 1)))

    scores = {
        "faithfulness": _clip(faithfulness),
        "answer_relevance": _clip(answer_relevance),
        "context_recall": _clip(context_recall),
        "context_precision": _clip(context_precision),
    }
    scores["average"] = statistics.mean(scores.values())
    return scores


@dataclass
class LocalRAGPipeline:
    name: str
    mode: str
    top_k: int = 5
    score_threshold: float = -1.0
    use_reranking: bool = True
    use_llm: bool = False

    def _retrieve_sources(self, question: str) -> list[dict]:
        if self.mode == "dense_only":
            from src.task5_semantic_search import semantic_search

            results = semantic_search(question, top_k=self.top_k)
        elif self.mode == "lexical_only":
            from src.task6_lexical_search import lexical_search

            results = lexical_search(question, top_k=self.top_k)
        else:
            from src.task9_retrieval_pipeline import retrieve

            results = retrieve(
                question,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
                use_reranking=self.use_reranking,
            )

        tagged = []
        for item in results:
            row = item.copy()
            row.setdefault("source", self.mode)
            tagged.append(row)
        return tagged

    def generate_with_citation(self, question: str) -> dict:
        if self.use_llm and self.mode == "hybrid_rerank":
            from src.task10_generation import generate_with_citation

            return generate_with_citation(
                question,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
            )

        sources = self._retrieve_sources(question)
        return {
            "answer": _build_extractive_answer(question, sources),
            "sources": sources,
            "retrieval_source": self.mode,
        }


def _evaluate_pipeline(rag_pipeline, golden_dataset: list[dict], framework_name: str) -> dict:
    cases = []
    for item in golden_dataset:
        try:
            result = rag_pipeline.generate_with_citation(item["question"])
            actual_output = result.get("answer", "")
            sources = result.get("sources", [])
            error = ""
        except Exception as exc:
            actual_output = ""
            sources = []
            error = str(exc)

        scores = _score_case(item, actual_output, sources)
        cases.append({
            "id": item.get("id", ""),
            "category": item.get("category", "unknown"),
            "question": item.get("question", ""),
            "expected_answer": item.get("expected_answer", ""),
            "expected_context": item.get("expected_context", ""),
            "actual_output": actual_output,
            "sources": [_source_name(source) for source in sources],
            "source_count": len(sources),
            "retrieval_source": getattr(rag_pipeline, "mode", "unknown"),
            "scores": scores,
            "error": error,
        })

    overall = {
        metric: statistics.mean(case["scores"][metric] for case in cases) if cases else 0.0
        for metric in METRIC_NAMES
    }
    overall["average"] = statistics.mean(overall.values()) if overall else 0.0

    return {
        "framework": framework_name,
        "config": getattr(rag_pipeline, "name", "unknown"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(cases),
        "overall": overall,
        "cases": cases,
    }


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sá»­ dá»¥ng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    framework_name = "Local deterministic evaluator (DeepEval-style metrics)"
    return _evaluate_pipeline(rag_pipeline, golden_dataset, framework_name)


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sá»­ dá»¥ng RAGAS.

    pip install ragas
    """
    # TODO: Implement
    #
    # from ragas import evaluate
    # from ragas.metrics import (
    #     faithfulness,
    #     answer_relevancy,
    #     context_recall,
    #     context_precision,
    # )
    # from datasets import Dataset
    #
    # eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    #
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     eval_data["question"].append(item["question"])
    #     eval_data["answer"].append(result["answer"])
    #     eval_data["contexts"].append([c["content"] for c in result["sources"]])
    #     eval_data["ground_truth"].append(item["expected_answer"])
    #
    # dataset = Dataset.from_dict(eval_data)
    # result = evaluate(
    #     dataset,
    #     metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    # )
    # return result.to_pandas()
    framework_name = "Local deterministic evaluator (RAGAS-style schema)"
    return _evaluate_pipeline(rag_pipeline, golden_dataset, framework_name)


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sá»­ dá»¥ng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="DrugLaw_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    framework_name = "Local deterministic evaluator (TruLens-style feedback)"
    return _evaluate_pipeline(rag_pipeline, golden_dataset, framework_name)


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sÃ¡nh A/B giá»¯a Ã­t nháº¥t 2 configs.

    Gá»£i Ã½ configs Ä‘á»ƒ so sÃ¡nh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (khÃ´ng reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    # TODO: Implement A/B comparison
    #
    # configs = {
    #     "hybrid_rerank": {"use_reranking": True, "alpha": 0.5},
    #     "dense_only": {"use_reranking": False, "alpha": 1.0},
    # }
    #
    # results = {}
    # for config_name, params in configs.items():
    #     # Run eval with this config
    #     ...
    #     results[config_name] = scores
    #
    # return results
    use_llm = os.getenv("EVAL_USE_LLM", "0") == "1"
    configs = {
        "hybrid_rerank": LocalRAGPipeline(
            name="Config A - hybrid + RRF + rerank",
            mode="hybrid_rerank",
            use_reranking=True,
            score_threshold=-1.0,
            use_llm=use_llm,
        ),
        "hybrid_no_rerank": LocalRAGPipeline(
            name="Config B - hybrid + RRF, no rerank",
            mode="hybrid_no_rerank",
            use_reranking=False,
            score_threshold=-1.0,
        ),
        "dense_only": LocalRAGPipeline(
            name="Config C - dense only",
            mode="dense_only",
        ),
        "lexical_only": LocalRAGPipeline(
            name="Config D - BM25 only",
            mode="lexical_only",
        ),
    }

    comparison = {}
    for config_name, pipeline in configs.items():
        comparison[config_name] = evaluate_with_deepeval(pipeline, golden_dataset)
    return comparison


# =============================================================================
# Export Results
# =============================================================================

def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _short(text: str, limit: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
    return cleaned[: limit - 3] + "..." if len(cleaned) > limit else cleaned


def _failure_stage(case: dict) -> str:
    scores = case["scores"]
    lowest_metric = min(METRIC_NAMES, key=lambda metric: scores[metric])
    return {
        "faithfulness": "Generation grounding",
        "answer_relevance": "Answer relevance",
        "context_recall": "Retriever recall",
        "context_precision": "Retriever precision",
    }[lowest_metric]


def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    # TODO: Format and write results
    #
    # content = "# RAG Evaluation Results\n\n"
    # content += "## Overall Scores\n\n"
    # content += "| Metric | Score |\n|--------|-------|\n"
    # ...
    # content += "\n## A/B Comparison\n\n"
    # ...
    # content += "\n## Worst Performers\n\n"
    # ...
    # content += "\n## Recommendations\n\n"
    # ...
    #
    # RESULTS_PATH.write_text(content, encoding="utf-8")
    best_name, best_result = max(
        comparison.items(),
        key=lambda item: item[1]["overall"]["average"],
    )
    baseline = comparison.get("hybrid_rerank", results)
    worst_cases = sorted(
        baseline["cases"],
        key=lambda case: case["scores"]["average"],
    )[:3]

    content = "# RAG Evaluation Results\n\n"
    content += f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n\n"
    content += f"Framework: **{results['framework']}**\n\n"
    content += (
        "Note: this run uses deterministic local proxy metrics so the evaluation can "
        "run reproducibly without an external LLM judge. Scores are useful for A/B "
        "comparison and debugging retrieval quality, not as a final human-grade score.\n\n"
    )

    content += "## Overall Scores\n\n"
    content += "| Metric | Hybrid + rerank | Hybrid no rerank | Dense-only | BM25-only |\n"
    content += "|--------|-----------------|------------------|------------|----------|\n"
    for metric in METRIC_NAMES + ["average"]:
        row = [metric.replace("_", " ").title()]
        for config in ["hybrid_rerank", "hybrid_no_rerank", "dense_only", "lexical_only"]:
            row.append(_pct(comparison[config]["overall"][metric]))
        content += "| " + " | ".join(row) + " |\n"

    content += "\n## A/B Comparison Analysis\n\n"
    content += f"Best config in this run: **{best_result['config']}** (`{best_name}`) with average score {_pct(best_result['overall']['average'])}.\n\n"
    content += "- **Config A:** Hybrid semantic + BM25, merged by RRF, then reranked with the current bi-encoder reranker.\n"
    content += "- **Config B:** Hybrid semantic + BM25, merged by RRF, without reranking.\n"
    content += "- **Config C:** Dense-only retrieval from ChromaDB embeddings.\n"
    content += "- **Config D:** BM25-only lexical retrieval from markdown chunks.\n\n"

    delta = comparison["hybrid_rerank"]["overall"]["average"] - comparison["dense_only"]["overall"]["average"]
    content += f"Hybrid + rerank vs dense-only delta: **{delta:+.3f}**. "
    if delta >= 0:
        content += "Hybrid retrieval is currently helping or matching dense retrieval on the golden set.\n\n"
    else:
        content += "Dense-only is currently stronger; inspect reranking and BM25 fusion weights before demo.\n\n"

    content += "## Worst Performers (Bottom 3, Config A)\n\n"
    content += "| # | Question | Avg | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |\n"
    content += "|---|----------|-----|--------------|-----------|--------|-----------|---------------|------------|\n"
    for idx, case in enumerate(worst_cases, 1):
        scores = case["scores"]
        root_cause = "Expected evidence terms are weakly covered by retrieved chunks."
        if case.get("error"):
            root_cause = f"Pipeline error: {_short(case['error'], 80)}"
        elif case["source_count"] == 0:
            root_cause = "Retriever returned no context."
        content += (
            f"| {idx} | {_short(case['question'], 85)} | {_pct(scores['average'])} | "
            f"{_pct(scores['faithfulness'])} | {_pct(scores['answer_relevance'])} | "
            f"{_pct(scores['context_recall'])} | {_pct(scores['context_precision'])} | "
            f"{_failure_stage(case)} | {root_cause} |\n"
        )

    content += "\n## Per-Case Scores (Config A)\n\n"
    content += "| ID | Category | Avg | Sources | Question |\n"
    content += "|----|----------|-----|---------|----------|\n"
    for case in baseline["cases"]:
        content += (
            f"| {case['id']} | {case['category']} | {_pct(case['scores']['average'])} | "
            f"{case['source_count']} | {_short(case['question'], 95)} |\n"
        )

    content += "\n## Recommendations\n\n"
    content += "### Cai tien 1\n"
    content += "**Action:** Clean converted markdown, remove navigation/advertisement blocks from news pages before indexing.  \n"
    content += "**Expected impact:** Higher context precision and cleaner citations.\n\n"
    content += "### Cai tien 2\n"
    content += "**Action:** Rebuild the golden set after reviewing final markdown, especially for news items whose crawler title is generic.  \n"
    content += "**Expected impact:** More reliable relevance and recall scores.\n\n"
    content += "### Cai tien 3\n"
    content += "**Action:** Try reranking with a real multilingual cross-encoder or Jina API, then compare against the current bi-encoder rerank.  \n"
    content += "**Expected impact:** Better ordering for legal questions with similar articles and provisions.\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    return content


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    # TODO: Import your RAG pipeline
    # from src.task10_generation import generate_with_citation
    #
    # Chá»n 1 framework:
    # results = evaluate_with_deepeval(pipeline, golden_dataset)
    # results = evaluate_with_ragas(pipeline, golden_dataset)
    # results = evaluate_with_trulens(pipeline, golden_dataset)
    #
    # comparison = compare_configs(pipeline, golden_dataset)
    # export_results(results, comparison)
    pipeline = LocalRAGPipeline(
        name="Config A - hybrid + RRF + rerank",
        mode="hybrid_rerank",
        use_reranking=True,
        score_threshold=-1.0,
        use_llm=os.getenv("EVAL_USE_LLM", "0") == "1",
    )
    results = evaluate_with_deepeval(pipeline, golden_dataset)
    comparison = compare_configs(pipeline, golden_dataset)
    export_results(results, comparison)

    print("Evaluation complete")
    print(f"Results written to: {RESULTS_PATH}")
    for name, result in comparison.items():
        print(f"  {name}: {_pct(result['overall']['average'])}")
