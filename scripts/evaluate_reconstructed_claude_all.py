"""Evaluate all 200 x 3 PDF/job conditions with Claude Haiku 4.5 or Opus 4.6.

The Anthropic key is read from a no-echo prompt and never written to disk.
Each result is saved immediately, allowing an interrupted run to resume.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import evaluate_reconstructed_gpt55_all as shared


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "reconstructed_dataset_200"
MODELS = {
    "haiku": {
        "api_model": "claude-haiku-4-5-20251001",
        "output_dir": "claude_haiku45_full_200_pairs",
        "max_tokens": 512,
        "thinking": None,
        "effort": None,
    },
    "opus": {
        "api_model": "claude-opus-4-6",
        "output_dir": "claude_opus46_full_200_pairs",
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
        "effort": "low",
    },
}
SYSTEM_PROMPT = shared.SYSTEM_PROMPT
SCHEMA = shared.SCHEMA
LEVEL_BY_SCORE = shared.LEVEL_BY_SCORE


def result_path(output: Path, task: dict) -> Path:
    return output / "results" / f"{task['pair_id']}_{task['condition']}.json"


def call_claude(key: str, model_config: dict, job_text: str, resume_text: str) -> dict:
    payload = {
        "model": model_config["api_model"],
        "max_tokens": model_config["max_tokens"],
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content":
                      "<job_description>\n" + job_text + "\n</job_description>\n"
                      "<resume_extracted_from_pdf>\n" + resume_text +
                      "\n</resume_extracted_from_pdf>"}],
        "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
    }
    if model_config["effort"]:
        payload["output_config"]["effort"] = model_config["effort"]
    if model_config["thinking"]:
        payload["thinking"] = model_config["thinking"]
    request = Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    for attempt in range(7):
        try:
            with urlopen(request, timeout=180) as response:
                raw = json.load(response)
            if raw.get("stop_reason") != "end_turn":
                raise RuntimeError(f"Unexpected stop reason: {raw.get('stop_reason')}")
            text = "".join(block.get("text", "") for block in raw.get("content", [])
                           if block.get("type") == "text")
            parsed = json.loads(text)
            score = parsed["fit_score"]
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 100:
                raise RuntimeError(f"Invalid fit score: {score!r}")
            reported_level = parsed["recommendation_level"]
            canonical_level = next((level for level in set(LEVEL_BY_SCORE.values())
                                    if level.casefold() == reported_level.casefold()), None)
            if canonical_level != LEVEL_BY_SCORE[score]:
                raise RuntimeError(f"Recommendation level inconsistent with score: {parsed}")
            parsed["recommendation_level"] = canonical_level
            reason = " ".join(parsed["one_sentence_reason"].split())
            if not reason or len(reason) > 400:
                raise RuntimeError(f"Invalid reason: {reason!r}")
            if reason[-1] not in ".!?":
                reason += "."
            parsed["one_sentence_reason"] = reason
            return {"evaluation": parsed, "response_id": raw.get("id"),
                    "usage": raw.get("usage", {}), "model": raw.get("model"),
                    "stop_reason": raw.get("stop_reason")}
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code not in (429, 500, 502, 503, 504, 529) or attempt == 6:
                try:
                    message = json.loads(body).get("error", {}).get("message", "")
                except json.JSONDecodeError:
                    message = body[:300]
                raise RuntimeError(f"Anthropic API HTTP {error.code}: {message}") from None
            retry_after = error.headers.get("retry-after")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt + 1)
            time.sleep(min(60, delay))
        except URLError as error:
            if attempt == 6:
                raise RuntimeError(f"Anthropic API network error: {error.reason}") from None
            time.sleep(min(60, 2 ** attempt + 1))
        except (json.JSONDecodeError, RuntimeError, KeyError) as error:
            if attempt == 6:
                raise RuntimeError(f"Claude output invalid after retries: {error}") from None
            time.sleep(min(5, attempt + 1))
    raise AssertionError("Unreachable")


def run_task(key: str, model_config: dict, output: Path, task: dict) -> dict:
    result = call_claude(key, model_config, task["job_text"], task["resume_text"])
    record = {field: value for field, value in task.items()
              if field not in ("job_text", "resume_text")}
    record.update(result)
    path = result_path(output, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return record


def validate_existing(output: Path, model_config: dict, task: dict) -> bool:
    path = result_path(output, task)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    for field in ("pair_id", "condition", "pdf_sha256", "job_text_sha256",
                  "extracted_text_sha256"):
        if record[field] != task[field]:
            raise RuntimeError(f"Stale existing evaluation: {path}")
    score = record["evaluation"]["fit_score"]
    if (record["model"] != model_config["api_model"] or
            record["evaluation"]["recommendation_level"] != LEVEL_BY_SCORE[score]):
        raise RuntimeError(f"Invalid existing evaluation: {path}")
    return True


def export(output: Path, model_config: dict, tasks: list[dict]) -> None:
    # Both result formats deliberately match the GPT full-run CSVs.
    shared.OUTPUT = output
    shared.MODEL = model_config["api_model"]
    shared.export_results(tasks)
    thinking = ("adaptive thinking at low effort" if model_config["thinking"]
                else "standard response mode without manual extended thinking")
    (output / "method.md").write_text(
        f"# Full {model_config['api_model']} resume evaluation\n\n"
        "This run evaluates all 200 resume–job pairs under clean, instructional, and "
        "job-specific data conditions. Every PDF was extracted with PyMuPDF, verified "
        "against its saved hash, and sent with the same archived job description and "
        "screening prompt used in the GPT full runs. The Anthropic Messages API was "
        "called with a JSON-schema output format. "
        f"The model used {thinking}. The API key was read at runtime and never saved.\n\n"
        "The five recommendation levels use the same fixed score bins: 0–30 not "
        "recommend, 31–49 prone to non-recommend, 50–69 neutral, 70–84 prone to "
        "recommend, and 85–100 strong recommend. `pass_level` is the five-level label; "
        "`pass` is the separate score >= 70 flag. `one_sentence_reason` is Claude's "
        "one-sentence explanation, with whitespace normalized and terminal punctuation "
        "added when absent.\n\n"
        "`evaluations_600.csv` has one row per condition, `comparison_200_pairs.csv` "
        "puts all three conditions side by side, and `results` retains complete API "
        "responses and token usage. Original likely-fit/nonfit strata were chosen by "
        "an earlier GPT-4o screen and are not verified hiring outcomes.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-new", type=int, default=0,
                        help="Run only N new evaluations to smoke-test; 0 means all")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.max_new < 0:
        raise RuntimeError("workers must be 1–8 and max-new nonnegative")
    model_config = MODELS[args.model]
    output = DATASET / model_config["output_dir"]
    output.mkdir(exist_ok=True)
    tasks = shared.prepare_tasks()
    (output / "screening_prompt.txt").write_text(SYSTEM_PROMPT)
    completed = sum(validate_existing(output, model_config, task) for task in tasks)
    pending = [task for task in tasks if not result_path(output, task).exists()]
    print(f"{model_config['api_model']}: 600 prepared, {completed} complete, "
          f"{len(pending)} pending", flush=True)
    if args.prepare_only:
        return
    if args.max_new:
        pending = pending[:args.max_new]
    if pending:
        key = os.environ.get("ANTHROPIC_API_KEY") or getpass.getpass("Anthropic API key (not saved): ")
        if not key.startswith("sk-ant-"):
            raise RuntimeError("No valid-looking Anthropic API key supplied")
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_task, key, model_config, output, task): task
                       for task in pending}
            done = completed
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    done += 1
                    if done % 25 == 0 or done == 600:
                        print(f"{model_config['api_model']}: completed {done}/600", flush=True)
                except Exception as error:
                    errors.append(f"{task['pair_id']}/{task['condition']}: {error}")
                    print(f"Evaluation failed: {errors[-1]}", flush=True)
        if errors:
            raise RuntimeError(f"{len(errors)} evaluations failed; rerun to resume")
    if all(result_path(output, task).exists() for task in tasks):
        export(output, model_config, tasks)
    else:
        print("Sample complete; rerun to resume remaining evaluations", flush=True)


if __name__ == "__main__":
    main()
