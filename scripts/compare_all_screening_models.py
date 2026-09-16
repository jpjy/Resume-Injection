"""Compare the four full screening runs on exactly matched PDF/job inputs."""

import csv
import hashlib
import json
from pathlib import Path


DATASET = Path(__file__).resolve().parents[1] / "reconstructed_dataset_200"
RUNS = {
    "gpt4o": DATASET / "gpt4o_full_200_pairs",
    "gpt55": DATASET / "gpt55_full_200_pairs",
    "haiku45": DATASET / "claude_haiku45_full_200_pairs",
    "opus46_low": DATASET / "claude_opus46_full_200_pairs",
}


def main() -> None:
    prompt_hashes = {name: hashlib.sha256((folder / "screening_prompt.txt").read_bytes()).hexdigest()
                     for name, folder in RUNS.items()}
    if len(set(prompt_hashes.values())) != 1:
        raise RuntimeError(f"Screening prompts differ: {prompt_hashes}")
    datasets = {}
    for name, folder in RUNS.items():
        rows = list(csv.DictReader((folder / "evaluations_600.csv").open()))
        datasets[name] = {(r["pair_id"], r["condition"]): r for r in rows}
        if len(rows) != 600 or len(datasets[name]) != 600:
            raise RuntimeError(f"Expected 600 unique evaluations for {name}")
    keys = set(datasets["gpt4o"])
    if any(set(rows) != keys for rows in datasets.values()):
        raise RuntimeError("Runs do not have identical pair/condition keys")
    joined = []
    for key in sorted(keys):
        source = datasets["gpt4o"][key]
        for name, rows in datasets.items():
            r = rows[key]
            for field in ("jd_id", "source_id", "fit_group", "pdf_sha256",
                          "job_text_sha256", "extracted_text_sha256"):
                if r[field] != source[field]:
                    raise RuntimeError(f"Unmatched {field}: {name}/{key}")
        item = {
            "pair_id": source["pair_id"], "jd_id": source["jd_id"],
            "source_id": source["source_id"], "fit_group": source["fit_group"],
            "condition": source["condition"], "pdf_file": source["pdf_file"],
        }
        for name, rows in datasets.items():
            r = rows[key]
            item.update({
                f"{name}_score": int(r["fit_score"]),
                f"{name}_pass_level": r["pass_level"],
                f"{name}_pass": r["pass"],
                f"{name}_reason": r["one_sentence_reason"],
            })
        joined.append(item)
    with (DATASET / "all_models_comparison_600.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    summary = {
        "matched_pairs": 200, "matched_evaluations_per_model": 600,
        "prompt_sha256": next(iter(prompt_hashes.values())),
        "models": {},
    }
    for name, folder in RUNS.items():
        own = json.loads((folder / "summary.json").read_text())
        summary["models"][name] = {
            "api_model": own["model"],
            "conditions": own["conditions"],
            "mean_score_deltas": own["mean_score_deltas"],
            "pass_transitions": own["pass_transitions"],
        }
    (DATASET / "all_models_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Joined 600 exactly matched PDF/job conditions across four models")


if __name__ == "__main__":
    main()
