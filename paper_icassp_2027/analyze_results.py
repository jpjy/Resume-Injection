"""Recompute paper tables from the archived four-model paired comparison."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reconstructed_dataset_200"
OUT = Path(__file__).resolve().parent
MODELS = ["gpt4o", "gpt55", "haiku45", "opus46_low"]
NAMES = ["GPT-4o", "GPT-5.5", "Haiku 4.5", "Opus 4.6"]


def main() -> None:
    rows = list(csv.DictReader((DATA / "all_models_comparison_600.csv").open()))
    jobs = list(csv.DictReader((DATA / "job_descriptions_20/manifest.csv").open()))
    by = {(r["pair_id"], r["condition"]): r for r in rows}
    pairs = sorted({r["pair_id"] for r in rows})
    if len(rows) != 600 or len(pairs) != 200 or len(jobs) != 20:
        raise RuntimeError("Expected 600 evaluations, 200 pairs, 20 jobs")
    summary = {"models": {}, "jobs": []}
    for model, name in zip(MODELS, NAMES):
        pass_counts = {
            condition: sum(by[(p, condition)][model + "_pass"] == "True" for p in pairs)
            for condition in ("clean", "instruction", "data")
        }
        transitions = {}
        for condition in ("instruction", "data"):
            transitions[condition] = {
                "fail_to_pass": sum(
                    by[(p, "clean")][model + "_pass"] == "False"
                    and by[(p, condition)][model + "_pass"] == "True"
                    for p in pairs
                ),
                "pass_to_fail": sum(
                    by[(p, "clean")][model + "_pass"] == "True"
                    and by[(p, condition)][model + "_pass"] == "False"
                    for p in pairs
                ),
                "mean_score_change": sum(
                    int(by[(p, condition)][model + "_score"])
                    - int(by[(p, "clean")][model + "_score"])
                    for p in pairs
                ) / len(pairs),
            }
        thresholds = {
            str(threshold): {
                condition: sum(
                    int(by[(p, condition)][model + "_score"]) >= threshold
                    for p in pairs
                )
                for condition in ("clean", "instruction", "data")
            }
            for threshold in (60, 70, 80, 85)
        }
        summary["models"][name] = {
            "passes": pass_counts, "transitions": transitions,
            "threshold_counts": thresholds,
        }
    for job in jobs:
        job_pairs = [p for p in pairs if p.startswith(job["jd_id"] + "_")]
        flips = {
            name: sum(
                by[(p, "clean")][model + "_pass"] == "False"
                and by[(p, "data")][model + "_pass"] == "True"
                for p in job_pairs
            )
            for model, name in zip(MODELS, NAMES)
        }
        summary["jobs"].append({
            "jd_id": job["jd_id"], "title": job["title"],
            "data_fail_to_pass": flips, "total_data_flips": sum(flips.values()),
        })
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "job_results.csv").open("w", newline="") as stream:
        fields = ["jd_id", "title", "gpt4o", "gpt55", "haiku45", "opus46_low", "total"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for job in summary["jobs"]:
            flips = job["data_fail_to_pass"]
            writer.writerow({
                "jd_id": job["jd_id"], "title": job["title"],
                "gpt4o": flips["GPT-4o"], "gpt55": flips["GPT-5.5"],
                "haiku45": flips["Haiku 4.5"], "opus46_low": flips["Opus 4.6"],
                "total": job["total_data_flips"],
            })
    ordered = sorted(summary["jobs"], key=lambda j: (-j["total_data_flips"], j["jd_id"]))
    lines = [
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Posting & GPT-4o & GPT-5.5 & Haiku & Opus & All \\",
        r"\midrule",
    ]
    for job in ordered:
        title = job["title"].replace("Representitive", "Representative")
        title = title.replace("&", r"\&").replace("_", r"\_")
        title = title[:37] + ("..." if len(title) > 37 else "")
        label = job["jd_id"].replace("_", r"\_") + " " + title
        values = [job["data_fail_to_pass"][name] for name in NAMES]
        cells = [rf"\cellcolor{{blue!{6 * value}}}{value}" for value in values]
        lines.append(label + " & " + " & ".join(cells) +
                     f" & {job['total_data_flips']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "job_heatmap.tex").write_text("\n".join(lines) + "\n")
    print("Verified four models, 20 postings, and 600 matched evaluations")


if __name__ == "__main__":
    main()
