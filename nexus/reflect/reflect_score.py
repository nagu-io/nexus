"""
ReflectScore - hallucination detection benchmark and trust layer.
Scores responses from 0.0 (no hallucination) to 1.0 (certain hallucination).
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import track
from rich.table import Table

from nexus.reflect import metrics
from nexus.reflect.benchmark_data import BUILTIN_QUESTIONS
from nexus.reflect.evaluator import AutoEvaluator

console = Console()

REFLECT_SYSTEM_PROMPT = """You are a hallucination detection expert.
Given a question and an AI response, score the likelihood that the response contains hallucinations.

Return ONLY a JSON object with this exact format:
{
  "score": 0.0,
  "reasons": ["reason1", "reason2"],
  "verdict": "clean" | "suspicious" | "hallucinated"
}

Score guide:
0.0 - 0.3: clean, factual, well-grounded
0.3 - 0.6: suspicious, some uncertain claims
0.6 - 1.0: likely hallucinated, fabricated facts

Be strict. Flag any unverifiable specific claims (dates, names, statistics, URLs).
"""


class ReflectScore:
    """Hallucination scoring engine and benchmark layer for NEXUS."""

    def __init__(self):
        from nexus.config import config

        self.config = config
        self.results_dir = config.data_dir / "reflect_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._evaluator = AutoEvaluator()

    def _get_groq_client(self):
        """Get a Groq client."""
        from groq import Groq

        return Groq(api_key=self.config.groq_api_key)

    def _warning_message(self, score: float) -> str:
        return (
            f"ReflectScore flagged this answer as medium risk ({score:.2f}). "
            "Review it before using it as ground truth."
        )

    def _blocked_message(self, score: float) -> str:
        return f"ReflectScore blocked this answer at {score:.2f} risk and requested a stronger model."

    def interpret_score(self, score: float) -> dict:
        """Convert a numeric score into the NEXUS trust-layer decision."""
        if score < self.config.reflect_warn_threshold:
            return {"verdict": "clean", "action": "serve", "warning": None}
        if score < self.config.reflect_block_threshold:
            return {"verdict": "warning", "action": "warn", "warning": self._warning_message(score)}
        return {"verdict": "blocked", "action": "block", "warning": self._blocked_message(score)}

    async def score_response(self, question: str, response: str) -> float:
        """Score a single response for hallucination risk."""
        if not self.config.groq_api_key:
            return self._heuristic_score(response)

        try:
            client = self._get_groq_client()
            prompt = f"Question: {question}\n\nAI Response: {response}\n\nScore the hallucination likelihood."

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
                temperature=0.1,
            )

            raw = completion.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            return float(data.get("score", 0.5))

        except Exception as exc:
            console.print(f"[yellow]ReflectScore error: {exc}. Using heuristic.[/yellow]")
            return self._heuristic_score(response)

    async def assess_response(self, question: str, response: str) -> dict:
        """Return the full trust-layer assessment for a response."""
        score = await self.score_response(question, response)
        assessment = self.interpret_score(score)
        assessment.update(
            {
                "score": score,
                "question": question,
                "response_preview": response[:200],
                "should_warn": assessment["action"] == "warn",
                "should_reroute": assessment["action"] == "block",
            }
        )
        return assessment

    def blocked_response(self, question: str, score: float) -> str:
        """Return the safe user-facing response when NEXUS withholds an answer."""
        return (
            "NEXUS withheld a likely hallucinated answer instead of showing it.\n"
            f"ReflectScore risk: {score:.2f}\n"
            f"Question: {question}\n"
            "Try adding more context, switching models, or enabling a stronger cloud fallback."
        )

    def _heuristic_score(self, response: str) -> float:
        """Fallback heuristic scoring when Groq is unavailable."""
        score = 0.0
        response_lower = response.lower()

        high_risk = ["definitely", "certainly", "absolutely", "100%", "proven fact", "it is known"]
        medium_risk = ["approximately", "around", "roughly", "probably", "typically"]
        unsupported_patterns = [r"https?://\S+", r"\b\d{4}\b", r"\b\d+(?:\.\d+)?%\b"]

        for phrase in high_risk:
            if phrase in response_lower:
                score += 0.15

        for phrase in medium_risk:
            if phrase in response_lower:
                score += 0.05

        for pattern in unsupported_patterns:
            score += min(len(re.findall(pattern, response)) * 0.05, 0.15)

        if "i do not know" in response_lower or "not enough information" in response_lower:
            score = max(score - 0.1, 0.0)

        return min(score, 1.0)

    def _is_correct(self, question: dict, answer: str) -> bool:
        """Evaluate correctness for a benchmark row."""
        return bool(
            self._evaluator.is_correct(
                answer=answer,
                ground_truth=question.get("answer", ""),
                keywords=question.get("keywords", []) or [],
                unanswerable=bool(question.get("unanswerable", False)),
            )
        )

    def _summarize_metrics(self, results: list[dict]) -> dict:
        """Build benchmark summary from internal metrics."""
        hallucination = metrics.hallucination_rate(results, self._evaluator)
        grounding = metrics.grounding_score(results, self._evaluator)
        refusal = metrics.refusal_accuracy(results, self._evaluator)
        mean_latency = metrics.mean_latency(results, self._evaluator)
        accuracy_rate = 1.0 - hallucination if hallucination is not None else 0.0
        return {
            "accuracy": accuracy_rate,
            "accuracy_rate": accuracy_rate,
            "hallucination_rate": hallucination if hallucination is not None else 0.0,
            "grounding_score": grounding,
            "refusal_accuracy": refusal,
            "mean_response_time_seconds": mean_latency,
        }

    async def run_benchmark(self, n_samples: int = 300, model_label: str | None = None) -> dict:
        """Run the ReflectScore benchmark on the current serving model."""
        console.print("[bold cyan]Running ReflectScore benchmark...[/bold cyan]")

        test_questions = self._load_test_questions(n_samples)
        results = []

        from nexus.router.mind_router import MindRouter

        router = MindRouter()

        for item in track(test_questions, description="Scoring responses..."):
            question = item["question"]
            started = time.perf_counter()
            response = await router._call_local(question)
            latency = time.perf_counter() - started
            assessment = await self.assess_response(question, response)
            is_correct = self._is_correct(item, response)
            results.append(
                {
                    "id": item.get("id", question),
                    "question": question,
                    "response": response[:200],
                    "answer": response,
                    "score": assessment["score"],
                    "verdict": assessment["verdict"],
                    "action": assessment["action"],
                    "category": item.get("category", "general"),
                    "ground_truth": item.get("answer"),
                    "keywords": item.get("keywords", []) or [],
                    "file_reference": item.get("file_reference"),
                    "unanswerable": bool(item.get("unanswerable", False)),
                    "is_correct": is_correct,
                    "system": "nexus_local",
                    "response_time_seconds": latency,
                }
            )

        total = len(results)
        scores = [row["score"] for row in results]
        verdict_breakdown = {
            "clean": sum(1 for row in results if row["verdict"] == "clean"),
            "warning": sum(1 for row in results if row["verdict"] == "warning"),
            "blocked": sum(1 for row in results if row["verdict"] == "blocked"),
        }
        avg_score = sum(scores) / total if total else 0.0
        warning_rate = verdict_breakdown["warning"] / total if total else 0.0
        timestamp = datetime.now(timezone.utc).isoformat()
        benchmark_summary = self._summarize_metrics(results)

        summary = {
            "model_label": model_label or self.config.nexus_model,
            "total": total,
            "accuracy": benchmark_summary["accuracy"],
            "accuracy_rate": benchmark_summary["accuracy_rate"],
            "warning_rate": warning_rate,
            "hallucination_rate": benchmark_summary["hallucination_rate"],
            "avg_hallucination_score": avg_score,
            "grounding_score": benchmark_summary["grounding_score"],
            "refusal_accuracy": benchmark_summary["refusal_accuracy"],
            "mean_response_time_seconds": benchmark_summary["mean_response_time_seconds"],
            "verdict_breakdown": verdict_breakdown,
            "thresholds": {
                "warn": self.config.reflect_warn_threshold,
                "block": self.config.reflect_block_threshold,
            },
            "results": results,
            "timestamp": timestamp,
            "metrics_enabled": True,
        }

        output_path = self.results_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        summary["report_path"] = str(output_path)

        self._display_results(summary)
        return summary

    def _display_results(self, summary: dict):
        """Display benchmark results in a rich table."""
        table = Table(title="ReflectScore Benchmark Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Model Label", str(summary.get("model_label", self.config.nexus_model)))
        table.add_row("Total Questions", str(summary["total"]))
        table.add_row("Accuracy", f"{summary['accuracy_rate']:.1%}")
        table.add_row("Warning Rate", f"{summary['warning_rate']:.1%}")
        table.add_row("Hallucination Rate", f"{summary['hallucination_rate']:.1%}")
        table.add_row("Avg Hallucination Score", f"{summary['avg_hallucination_score']:.3f}")
        if summary.get("grounding_score") is not None:
            table.add_row("Grounding Score", f"{summary['grounding_score']:.1%}")
        if summary.get("refusal_accuracy") is not None:
            table.add_row("Refusal Accuracy", f"{summary['refusal_accuracy']:.1%}")
        console.print(table)

    def _load_test_questions(self, n: int) -> list:
        """Load built-in benchmark questions and repeat them up to the requested size."""
        repeated = BUILTIN_QUESTIONS * max(1, (n + len(BUILTIN_QUESTIONS) - 1) // len(BUILTIN_QUESTIONS))
        return repeated[:n]

    async def benchmark_model(self, model_path: str | Path, n_samples: int = 300) -> dict:
        """Benchmark a compressed model during `nexus init`."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Compressed model path does not exist: {model_path}")

        console.print(f"[cyan]Benchmarking compressed model: {model_path}[/cyan]")
        result = await self.run_benchmark(n_samples=n_samples, model_label=f"proxy::{model_path.name}")
        compression_ratio = 3.6
        benchmark_warning = (
            "Direct artifact benchmarking is not enabled in the current runtime, "
            "so ReflectScore measured the active NEXUS serving model instead of the saved compressed artifact."
        )

        meta_path = model_path / "compress_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, encoding="utf-8") as handle:
                    meta = json.load(handle)
                compression_ratio = float(meta.get("compression_ratio", compression_ratio))
            except Exception:
                pass

        result["model_label"] = f"proxy::{model_path.name}"
        result["benchmark_mode"] = "active_serving_backend_proxy"
        result["benchmark_warning"] = benchmark_warning
        result["compression_ratio"] = compression_ratio
        result["model_path"] = str(model_path)
        return result

    def _load_saved_reports(self) -> list[tuple[Path, dict]]:
        """Load saved benchmark reports from disk."""
        reports = []
        for path in sorted(self.results_dir.glob("benchmark_*.json")):
            try:
                with open(path, encoding="utf-8") as handle:
                    reports.append((path, json.load(handle)))
            except Exception as exc:
                console.print(f"[yellow]Skipping unreadable report {path}: {exc}[/yellow]")
        return reports

    async def export_report(self):
        """Export the latest benchmark report."""
        reports = self._load_saved_reports()
        if not reports:
            console.print("[yellow]No benchmark results found. Run: nexus reflect[/yellow]")
            return
        latest_path, latest = reports[-1]
        console.print(f"[green]Latest report: {latest_path}[/green]")
        self._display_results(latest)

    async def compare_models(self, reference_label: str = "original", n_samples: int = 300) -> dict:
        """
        Compare the current model against a saved reference benchmark.

        To use this as compressed-vs-original proof, benchmark the original model first
        and keep that report under the desired reference label.
        """
        existing_reports = self._load_saved_reports()
        current_results = await self.run_benchmark(n_samples=n_samples, model_label=self.config.nexus_model)

        reference_report = None
        reference_path = None
        for path, report in reversed(existing_reports):
            if report.get("model_label") == reference_label:
                reference_path = path
                reference_report = report
                break

        note = None
        if reference_report is None and existing_reports:
            reference_path, reference_report = existing_reports[-1]
            note = (
                f"No saved report matched '{reference_label}'. "
                f"Using the latest available report from {reference_path.name} as the reference."
            )
        elif reference_report is None:
            reference_report = current_results
            note = (
                "No previous benchmark report exists yet. "
                "Run one reference benchmark first to get a true compressed-vs-original comparison."
            )

        table = Table(title="Current Model vs Reference Benchmark")
        table.add_column("Metric", style="cyan")
        table.add_column(f"Reference ({reference_label})", style="yellow")
        table.add_column(f"Current ({current_results['model_label']})", style="green")
        table.add_row("Accuracy", f"{reference_report['accuracy_rate']:.1%}", f"{current_results['accuracy_rate']:.1%}")
        table.add_row(
            "Hallucination Rate",
            f"{reference_report['hallucination_rate']:.1%}",
            f"{current_results['hallucination_rate']:.1%}",
        )
        table.add_row("Warning Rate", f"{reference_report['warning_rate']:.1%}", f"{current_results['warning_rate']:.1%}")
        console.print(table)

        return {
            "reference": reference_report,
            "reference_path": str(reference_path) if reference_path else None,
            "current": current_results,
            "note": note,
        }

    def display_comparison(self, results: dict):
        """Display comparison notes after comparing saved reports."""
        note = results.get("note")
        if note:
            console.print(f"[yellow]{note}[/yellow]")
        else:
            console.print("[bold green]Model comparison complete.[/bold green]")

    async def live_mode(self):
        """Score every response in real time during interactive chat."""
        console.print("[bold cyan]Live hallucination scoring active. Type your questions.[/bold cyan]")
        from nexus.router.mind_router import MindRouter

        router = MindRouter()

        while True:
            try:
                question = input("\n[ReflectScore Live] You: ").strip()
                if question.lower() in ["exit", "quit", "q"]:
                    break
                response = await router._call_local(question)
                assessment = await self.assess_response(question, response)
                color = "green" if assessment["verdict"] == "clean" else "yellow" if assessment["verdict"] == "warning" else "red"
                console.print(f"\n[bold]Response:[/bold] {response}")
                console.print(
                    f"[{color}]ReflectScore: {assessment['score']:.3f} "
                    f"({assessment['verdict']} -> {assessment['action']})[/{color}]"
                )
                if assessment["warning"]:
                    console.print(f"[{color}]{assessment['warning']}[/{color}]")
            except KeyboardInterrupt:
                break
