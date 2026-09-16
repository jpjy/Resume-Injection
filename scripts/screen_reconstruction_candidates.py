"""Screen shortlisted original PDF text against revised job descriptions."""

import argparse
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import zipfile

import pymupdf

from evaluate_gpt4o_pilot import api_call, sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reconstructed_dataset_200"
ARCHIVE = ROOT / "source_resumes_300" / "source_archive.zip"
JOBS = OUTPUT / "job_descriptions_20"
ALLOWED_CATEGORIES = {
    "jd_01": {"DESIGNER", "DIGITAL-MEDIA", "ARTS", "PUBLIC-RELATIONS"},
    "jd_02": {"CONSULTANT", "BUSINESS-DEVELOPMENT", "FINANCE", "HR"},
    "jd_03": {"HR"},
    "jd_04": {"ACCOUNTANT", "FINANCE"},
    "jd_05": {"HR"},
    "jd_06": {"SALES", "BUSINESS-DEVELOPMENT"},
    "jd_07": {"ACCOUNTANT", "FINANCE"},
    "jd_08": {"FINANCE", "BANKING", "ACCOUNTANT"},
    "jd_09": {"DIGITAL-MEDIA", "PUBLIC-RELATIONS", "BUSINESS-DEVELOPMENT"},
    "jd_10": {"DIGITAL-MEDIA", "PUBLIC-RELATIONS", "BUSINESS-DEVELOPMENT"},
    "jd_11": {"DIGITAL-MEDIA", "PUBLIC-RELATIONS", "BUSINESS-DEVELOPMENT"},
    "jd_12": {"HR"},
    "jd_13": {"DESIGNER", "ARTS", "DIGITAL-MEDIA", "PUBLIC-RELATIONS"},
    "jd_14": {"DESIGNER", "ARTS", "DIGITAL-MEDIA", "PUBLIC-RELATIONS"},
    "jd_15": {"HR"},
    "jd_16": {"SALES", "BUSINESS-DEVELOPMENT", "BANKING"},
    "jd_17": {"DIGITAL-MEDIA", "PUBLIC-RELATIONS", "BUSINESS-DEVELOPMENT",
              "APPAREL", "ARTS", "SALES"},
    "jd_18": {"SALES", "BUSINESS-DEVELOPMENT"},
    "jd_19": {"BPO", "SALES", "BANKING", "CONSULTANT", "ADVOCATE",
              "HEALTHCARE", "AVIATION", "APPAREL", "PUBLIC-RELATIONS",
              "DIGITAL-MEDIA"},
    "jd_20": {"HR"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", default="all", help="all or comma-separated jd IDs")
    parser.add_argument("--limit", type=int, default=12,
                        help="Top N lexical candidates within plausible categories")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    selected_jobs = set(ALLOWED_CATEGORIES) if args.jobs == "all" else set(args.jobs.split(","))
    if not selected_jobs <= set(ALLOWED_CATEGORIES):
        raise RuntimeError(f"Unknown job IDs: {selected_jobs - set(ALLOWED_CATEGORIES)}")
    rankings = list(csv.DictReader((OUTPUT / "rankings.csv").open()))
    job_rows = {r["jd_id"]: r for r in csv.DictReader((JOBS / "manifest.csv").open())}
    tasks = []
    for jd_id in sorted(selected_jobs):
        candidates = [r for r in rankings if r["jd_id"] == jd_id
                      and r["source_category"] in ALLOWED_CATEGORIES[jd_id]]
        candidates.sort(key=lambda r: int(r["rank"]))
        tasks.extend(candidates[:args.limit])
    if args.prepare_only:
        for jd_id in sorted(selected_jobs):
            rows = [r for r in tasks if r["jd_id"] == jd_id]
            print(jd_id, [(r["rank"], r["source_category"], r["source_id"]) for r in rows])
        print(f"Prepared {len(tasks)} clean-PDF screening tasks; no API calls made.")
        return
    key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key (not saved): ")
    if not key.startswith("sk-"):
        raise RuntimeError("No valid-looking API key supplied")
    results_dir = OUTPUT / "candidate_screening"
    texts_dir = OUTPUT / "candidate_text"
    results_dir.mkdir(exist_ok=True)
    texts_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for index, candidate in enumerate(tasks, 1):
            jd_id = candidate["jd_id"]
            source_id = candidate["source_id"]
            result_path = results_dir / f"{jd_id}_{source_id}.json"
            if result_path.exists():
                continue
            pdf_data = archive.read(candidate["archive_entry"])
            if not pdf_data.startswith(b"%PDF"):
                continue
            try:
                document = pymupdf.open(stream=pdf_data, filetype="pdf")
                text = "\n".join(page.get_text() for page in document)
                pages = len(document)
            except Exception as error:
                print(f"{jd_id} {source_id}: PDF extraction failed: {error}", flush=True)
                continue
            if len(text.strip()) < 250:
                print(f"{jd_id} {source_id}: PDF text too short", flush=True)
                continue
            text_path = texts_dir / f"{source_id}.txt"
            text_path.write_text(text)
            jd_text = (JOBS / job_rows[jd_id]["text_file"]).read_text()
            result = api_call(key, jd_text, text)
            record = {
                "jd_id": jd_id, "source_id": source_id,
                "source_category": candidate["source_category"],
                "archive_entry": candidate["archive_entry"],
                "lexical_rank": int(candidate["rank"]),
                "lexical_score": float(candidate["lexical_score"]),
                "pdf_sha256": sha256(pdf_data),
                "extracted_text_sha256": sha256(text.encode()),
                "job_text_sha256": sha256(jd_text.encode()),
                "source_pages": pages,
                **result,
            }
            result_path.write_text(json.dumps(record, indent=2) + "\n")
            score = result["evaluation"]["fit_score"]
            decision = result["evaluation"]["decision"]
            print(f"{index}/{len(tasks)} {jd_id} {source_id}: {score} {decision}", flush=True)
    print(f"Screened candidate results in {results_dir}")


if __name__ == "__main__":
    main()
