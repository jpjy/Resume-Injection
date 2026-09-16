"""Paired GPT-4o pilot on 10 revised clean resume–job pairs and both injections."""

import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import random

import pymupdf

from evaluate_gpt4o_pilot import api_call, MODEL


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "reconstructed_dataset_200"
OUTPUT = DATASET / "gpt4o_pilot_10_pairs"
SEED = 20260915


def choose_pairs(pairs: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    jobs = sorted({r["jd_id"] for r in pairs})
    chosen_jobs = sorted(rng.sample(jobs, 10))
    plan = []
    used_sources = set()
    for index, jd_id in enumerate(chosen_jobs):
        group = "likely_fit" if index < 5 else "likely_nonfit"
        eligible = [r for r in pairs if r["jd_id"] == jd_id and r["fit_group"] == group]
        rng.shuffle(eligible)
        selected = next((r for r in eligible if r["source_id"] not in used_sources), eligible[0])
        used_sources.add(selected["source_id"])
        plan.append(selected)
    return plan


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    pairs = list(csv.DictReader((DATASET / "manifest.csv").open()))
    injected = {(r["pair_id"], r["kind"]): r for r in
                csv.DictReader((DATASET / "injected_manifest.csv").open())}
    jobs = {r["jd_id"]: r for r in
            csv.DictReader((DATASET / "job_descriptions_20" / "manifest.csv").open())}
    plan = choose_pairs(pairs)
    if len(pairs) != 200 or len(injected) != 400 or len(plan) != 10:
        raise RuntimeError("Expected 200 clean pairs and 400 injection variants")
    key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key (not saved): ")
    if not key.startswith("sk-"):
        raise RuntimeError("No valid-looking API key supplied")
    rows = []
    for index, pair in enumerate(plan, 1):
        pair_id = pair["pair_id"]
        jd_id = pair["jd_id"]
        job_text = (DATASET / "job_descriptions_20" / jobs[jd_id]["text_file"]).read_text()
        baseline_path = DATASET / pair["baseline_result"]
        baseline = json.loads(baseline_path.read_text())
        clean_text = (DATASET / pair["clean_extracted_text"]).read_text()
        if hashlib.sha256(clean_text.encode()).hexdigest() != baseline["extracted_text_sha256"]:
            raise RuntimeError(f"Baseline clean text changed: {pair_id}")
        if baseline["model"] != MODEL or baseline["job_text_sha256"] != pair["job_text_sha256"]:
            raise RuntimeError(f"Baseline protocol or job changed: {pair_id}")
        evaluations = {"clean": baseline["evaluation"]}
        for kind in ("instruction", "data"):
            variant = injected[(pair_id, kind)]
            text_path = DATASET / variant["extracted_text"]
            pdf_path = DATASET / variant["injected_pdf"]
            if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != variant["injected_pdf_sha256"]:
                raise RuntimeError(f"Injected PDF changed: {pair_id}/{kind}")
            extracted = "\n".join(page.get_text() for page in pymupdf.open(pdf_path))
            if extracted != text_path.read_text():
                raise RuntimeError(f"Injected PDF extraction sidecar changed: {pair_id}/{kind}")
            result_path = OUTPUT / "results" / f"{pair_id}_{kind}.json"
            result_path.parent.mkdir(exist_ok=True)
            if result_path.exists():
                result = json.loads(result_path.read_text())
            else:
                result = api_call(key, job_text, extracted)
                result_path.write_text(json.dumps(result, indent=2) + "\n")
            evaluations[kind] = result["evaluation"]
            print(f"{index}/10 {pair_id} {kind}: {result['evaluation']['fit_score']} {result['evaluation']['decision']}", flush=True)
        rows.append({
            "pair_id": pair_id, "jd_id": jd_id,
            "job_title": pair["job_title"], "job_url": pair["job_url"],
            "source_id": pair["source_id"], "fit_group": pair["fit_group"],
            "clean_pdf": pair["clean_pdf"],
            "instruction_pdf": injected[(pair_id, "instruction")]["injected_pdf"],
            "data_pdf": injected[(pair_id, "data")]["injected_pdf"],
            "clean_score": evaluations["clean"]["fit_score"],
            "clean_decision": evaluations["clean"]["decision"],
            "instruction_score": evaluations["instruction"]["fit_score"],
            "instruction_decision": evaluations["instruction"]["decision"],
            "data_score": evaluations["data"]["fit_score"],
            "data_decision": evaluations["data"]["decision"],
            "instruction_delta": evaluations["instruction"]["fit_score"] - evaluations["clean"]["fit_score"],
            "data_delta": evaluations["data"]["fit_score"] - evaluations["clean"]["fit_score"],
        })
    with (OUTPUT / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "model": MODEL, "seed": SEED, "pairs": len(rows),
        "fit_groups": {g: sum(r["fit_group"] == g for r in rows)
                       for g in ("likely_fit", "likely_nonfit")},
        "mean_scores": {k: sum(int(r[f"{k}_score"]) for r in rows) / len(rows)
                        for k in ("clean", "instruction", "data")},
        "passes": {k: sum(r[f"{k}_decision"] == "pass" for r in rows)
                   for k in ("clean", "instruction", "data")},
        "mean_deltas": {k: sum(int(r[f"{k}_delta"]) for r in rows) / len(rows)
                        for k in ("instruction", "data")},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
