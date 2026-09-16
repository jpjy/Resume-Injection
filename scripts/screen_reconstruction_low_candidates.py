"""Add evaluated nonmatch candidates where a job has fewer than five."""

import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import zipfile

import pymupdf

from evaluate_gpt4o_pilot import api_call, sha256
from screen_reconstruction_candidates import OUTPUT, ARCHIVE, JOBS, ALLOWED_CATEGORIES


def main() -> None:
    rankings = list(csv.DictReader((OUTPUT / "rankings.csv").open()))
    jobs = {r["jd_id"]: r for r in csv.DictReader((JOBS / "manifest.csv").open())}
    result_dir = OUTPUT / "candidate_screening"
    text_dir = OUTPUT / "candidate_text"
    result_dir.mkdir(exist_ok=True)
    text_dir.mkdir(exist_ok=True)
    key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key (not saved): ")
    if not key.startswith("sk-"):
        raise RuntimeError("No valid-looking API key supplied")
    with zipfile.ZipFile(ARCHIVE) as archive:
        for jd_id in sorted(jobs):
            existing = [json.loads(p.read_text()) for p in result_dir.glob(f"{jd_id}_*.json")]
            fails = {r["source_id"] for r in existing if r["evaluation"]["decision"] == "fail"}
            if len(fails) >= 5:
                continue
            seen = {r["source_id"] for r in existing}
            # Source PDFs from unrelated corpus categories are useful clear
            # nonmatches; the actual PDF and job are still screened together.
            candidates = [r for r in rankings if r["jd_id"] == jd_id
                          and r["source_category"] not in ALLOWED_CATEGORIES[jd_id]
                          and r["source_id"] not in seen]
            candidates.sort(key=lambda r: (-float(r["lexical_score"]), r["source_id"]))
            job_text = (JOBS / jobs[jd_id]["text_file"]).read_text()
            for candidate in candidates:
                if len(fails) >= 5:
                    break
                source_id = candidate["source_id"]
                pdf_data = archive.read(candidate["archive_entry"])
                if not pdf_data.startswith(b"%PDF"):
                    continue
                try:
                    doc = pymupdf.open(stream=pdf_data, filetype="pdf")
                    text = "\n".join(page.get_text() for page in doc)
                    pages = len(doc)
                except Exception:
                    continue
                if len(text.strip()) < 250:
                    continue
                (text_dir / f"{source_id}.txt").write_text(text)
                result = api_call(key, job_text, text)
                record = {
                    "jd_id": jd_id, "source_id": source_id,
                    "source_category": candidate["source_category"],
                    "archive_entry": candidate["archive_entry"],
                    "lexical_rank": int(candidate["rank"]),
                    "lexical_score": float(candidate["lexical_score"]),
                    "pdf_sha256": sha256(pdf_data),
                    "extracted_text_sha256": sha256(text.encode()),
                    "job_text_sha256": sha256(job_text.encode()),
                    "source_pages": pages, **result,
                }
                (result_dir / f"{jd_id}_{source_id}.json").write_text(json.dumps(record, indent=2) + "\n")
                score = result["evaluation"]["fit_score"]
                decision = result["evaluation"]["decision"]
                print(f"{jd_id} {source_id} {candidate['source_category']}: {score} {decision}", flush=True)
                if decision == "fail":
                    fails.add(source_id)
                seen.add(source_id)
            if len(fails) < 5:
                raise RuntimeError(f"Could not find five nonmatches for {jd_id}")
    print("All jobs have at least five evaluated nonmatches")


if __name__ == "__main__":
    main()
