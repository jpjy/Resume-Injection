"""Assemble 20 x (5 likely fits + 5 likely nonfits) original PDF pairs."""

import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reconstructed_dataset_200"
ARCHIVE = ROOT / "source_resumes_300" / "source_archive.zip"


def main() -> None:
    jobs = list(csv.DictReader((OUT / "job_descriptions_20" / "manifest.csv").open()))
    old_ids = {r["source_id"] for r in csv.DictReader((ROOT / "source_resumes_300" / "manifest.csv").open())}
    if len(jobs) != 20:
        raise RuntimeError("Expected 20 archived job descriptions")
    clean_dir = OUT / "clean_pdfs"
    clean_dir.mkdir(exist_ok=True)
    records = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for job in jobs:
            jd_id = job["jd_id"]
            candidates = [json.loads(p.read_text()) for p in
                          (OUT / "candidate_screening").glob(f"{jd_id}_*.json")]
            high = [r for r in candidates if r["evaluation"]["decision"] == "pass"
                    and r["evaluation"]["fit_score"] >= 70]
            low = [r for r in candidates if r["evaluation"]["decision"] == "fail"
                   and r["evaluation"]["fit_score"] < 70]
            high.sort(key=lambda r: (-r["evaluation"]["fit_score"],
                                     r["lexical_rank"], r["source_id"]))
            low.sort(key=lambda r: (-r["evaluation"]["fit_score"],
                                    r["lexical_rank"], r["source_id"]))
            if len(high) < 5 or len(low) < 5:
                raise RuntimeError(f"{jd_id}: {len(high)} fits and {len(low)} nonfits; need five each")
            if any(r["job_text_sha256"] != job["text_sha256"] for r in candidates):
                raise RuntimeError(f"Stale baseline screening results for {jd_id}")
            # Include clear mismatches alongside partial-fit near misses when
            # the evaluated candidate pool supports both difficulty levels.
            far = sorted((r for r in low if r["evaluation"]["fit_score"] <= 49),
                         key=lambda r: (r["evaluation"]["fit_score"], r["source_id"]))
            near = [r for r in low if r["evaluation"]["fit_score"] >= 50]
            chosen_low = (near[:3] + far[:2]) if len(near) >= 3 and len(far) >= 2 else low[:5]
            selected = [("likely_fit", r) for r in high[:5]] + [("likely_nonfit", r) for r in chosen_low]
            if len({r["source_id"] for _, r in selected}) != 10:
                raise RuntimeError(f"Source PDF repeated within {jd_id}")
            for position, (group, r) in enumerate(selected, 1):
                source_id = r["source_id"]
                if source_id in old_ids:
                    raise RuntimeError(f"Old 300-PDF sample source reused: {source_id}")
                data = archive.read(r["archive_entry"])
                digest = hashlib.sha256(data).hexdigest()
                if digest != r["pdf_sha256"] or not data.startswith(b"%PDF"):
                    raise RuntimeError(f"Source PDF hash or signature mismatch: {source_id}")
                pair_id = f"{jd_id}_r{position:02d}"
                filename = f"{pair_id}_{source_id}.pdf"
                destination = clean_dir / jd_id / filename
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(data)
                text_path = OUT / "candidate_text" / f"{source_id}.txt"
                if not text_path.exists() or hashlib.sha256(text_path.read_bytes()).hexdigest() != r["extracted_text_sha256"]:
                    raise RuntimeError(f"Extracted source PDF text mismatch: {source_id}")
                records.append({
                    "pair_id": pair_id, "jd_id": jd_id,
                    "job_title": job["title"], "job_url": job["posting_url"],
                    "job_text_sha256": job["text_sha256"],
                    "fit_group": group,
                    "baseline_fit_score": r["evaluation"]["fit_score"],
                    "baseline_decision": r["evaluation"]["decision"],
                    "source_id": source_id, "source_category": r["source_category"],
                    "archive_entry": r["archive_entry"],
                    "clean_pdf": str(destination.relative_to(OUT)),
                    "clean_pdf_sha256": digest,
                    "clean_extracted_text": str(text_path.relative_to(OUT)),
                    "baseline_result": f"candidate_screening/{jd_id}_{source_id}.json",
                    "source_pages": r["source_pages"],
                    "lexical_rank": r["lexical_rank"],
                })
            print(f"{jd_id}: five fits + five nonfits", flush=True)
    if len(records) != 200:
        raise RuntimeError(f"Expected 200 pairs, got {len(records)}")
    with (OUT / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    distinct = len({r["source_id"] for r in records})
    (OUT / "README.md").write_text(
        "# Reconstructed resume–job dataset\n\n"
        "Twenty real job descriptions are archived in `job_descriptions_20`. Each job has five "
        "GPT-4o baseline likely-fit resume pairs and five likely-nonfit pairs. The label is a "
        "screening-model judgment at a 70-point threshold, not a verified hiring outcome.\n\n"
        "All clean PDFs are byte-identical copies of actual PDFs in the public Kaggle "
        "`snehaanbhawal/resume-dataset` archive, with hashes and archive entries in `manifest.csv`. "
        "The previous 300 source IDs are excluded. A source PDF may pair with multiple jobs, "
        "but every job uses ten different source IDs.\n\n"
        f"Pairs: 200. Distinct new source PDFs: {distinct}.\n"
    )
    print(f"Assembled 200 clean resume–job pairs from {distinct} distinct new source PDFs")


if __name__ == "__main__":
    main()
