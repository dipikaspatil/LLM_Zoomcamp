"""
Evaluation pipeline for the Knowledge Agent (RAG).

Measures two different things, since a RAG system can fail in two independent ways:
1. Retrieval quality — does search_rag() actually find the chunk containing
   the answer, and how high does it rank? (Hit Rate, MRR)
2. Answer quality — once the LLM generates an answer from whatever it
   retrieved, is it relevant and faithful to that context, or did it drift
   off-topic or invent something? (LLM-as-judge)

Run from the project root (capstone/) with:
    python eval/run_eval.py
"""
import sys
import pathlib

# eval/ sits next to backend/, not inside it — add backend/ to the import
# path so the "from app...." imports below (the real production code
# we're evaluating) resolve correctly regardless of cwd
BACKEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import asyncio
import json

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from app.config import settings
from app.tools.rag_search import search_rag
from app.agents.knowledge_agent import knowledge_agent_node

GROUND_TRUTH_PATH = pathlib.Path(__file__).resolve().parent / "ground_truth.jsonl"


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


async def evaluate_retrieval(ground_truth: list[dict]) -> dict:
    """
    Hit Rate: fraction of questions where the expected source document
    appears anywhere in the top-k retrieved chunks.
    MRR (Mean Reciprocal Rank): average of 1/rank of the expected source's
    first appearance (0 if never found) — rewards finding the right chunk
    near the top, not just somewhere in the top-k.
    """
    hits = 0
    reciprocal_ranks = []

    for item in ground_truth:
        chunks = await search_rag(item["question"], top_k=5)
        sources = [c["source"] for c in chunks]

        if item["expected_source"] in sources:
            hits += 1
            rank = sources.index(item["expected_source"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return {
        "hit_rate": hits / len(ground_truth),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
    }


class JudgeScore(BaseModel):
    relevance: int = Field(description="1-5: how directly the answer addresses the question")
    faithfulness: int = Field(description="1-5: how well the answer is grounded in the provided context, not hallucinated")
    reasoning: str = Field(description="One short sentence explaining the scores")


_judge_llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY, temperature=0)
_structured_judge = _judge_llm.with_structured_output(JudgeScore)


async def judge_answer(question: str, context: str, answer: str) -> JudgeScore:
    return await _structured_judge.ainvoke(
        "You are grading an AI assistant's answer for a football knowledge base.\n\n"
        f"Question: {question}\n\n"
        f"Context the assistant had access to:\n{context}\n\n"
        f"Assistant's answer: {answer}\n\n"
        "Score relevance (does it answer the question) and faithfulness "
        "(is it grounded in the context, not made up) from 1 (poor) to 5 (excellent)."
    )


async def evaluate_answers(ground_truth: list[dict]) -> dict:
    relevance_scores = []
    faithfulness_scores = []
    results = []

    for item in ground_truth:
        chunks = await search_rag(item["question"], top_k=5)
        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)

        agent_result = await knowledge_agent_node({"question": item["question"]})
        answer = agent_result["answer"]

        score = await judge_answer(item["question"], context, answer)
        relevance_scores.append(score.relevance)
        faithfulness_scores.append(score.faithfulness)

        results.append({
            "question": item["question"],
            "answer": answer,
            "relevance": score.relevance,
            "faithfulness": score.faithfulness,
            "reasoning": score.reasoning,
        })

    return {
        "avg_relevance": sum(relevance_scores) / len(relevance_scores),
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "results": results,
    }


async def main():
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground-truth questions\n")

    print("=== Retrieval Evaluation ===")
    retrieval_metrics = await evaluate_retrieval(ground_truth)
    print(f"Hit Rate: {retrieval_metrics['hit_rate']:.2%}")
    print(f"MRR:      {retrieval_metrics['mrr']:.3f}\n")

    print("=== Answer Quality Evaluation (LLM-as-judge) ===")
    answer_metrics = await evaluate_answers(ground_truth)
    print(f"Avg Relevance:    {answer_metrics['avg_relevance']:.2f} / 5")
    print(f"Avg Faithfulness: {answer_metrics['avg_faithfulness']:.2f} / 5\n")

    print("Per-question results:")
    for r in answer_metrics["results"]:
        print(f"- [{r['relevance']}/5 rel, {r['faithfulness']}/5 faith] {r['question']}")
        print(f"  {r['reasoning']}")


if __name__ == "__main__":
    asyncio.run(main())
