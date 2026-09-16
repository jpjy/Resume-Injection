"""Create instruction and job-specific data PDF variants for all 200 pairs."""

import csv
import hashlib
import json
from pathlib import Path
import shutil

from generate_font_injected_resumes import FONT_SOURCE, overlay_bytes, merge_resume, check_pdf


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "reconstructed_dataset_200"
OUTPUT = DATASET / "font_injected_pdfs"


def main() -> None:
    expected_font = "f8ace1f892b2bd9dc1792ba7f097fa7588f84fed48321480e04de5390828221f"
    if not FONT_SOURCE.exists():
        raise RuntimeError(f"Missing exact experiment font: {FONT_SOURCE}; see assets/README.md")
    if hashlib.sha256(FONT_SOURCE.read_bytes()).hexdigest() != expected_font:
        raise RuntimeError("Experiment font hash mismatch; see assets/README.md")
    pairs = list(csv.DictReader((DATASET / "manifest.csv").open()))
    jobs = list(csv.DictReader((DATASET / "job_descriptions_20" / "manifest.csv").open()))
    payloads = json.loads((DATASET / "injection_payloads.json").read_text())
    if len(pairs) != 200 or len(jobs) != 20:
        raise RuntimeError("Expected 200 clean pairs and 20 real job descriptions")
    if set(payloads["data_by_jd"]) != {r["jd_id"] for r in jobs}:
        raise RuntimeError("Every job needs its own data payload")
    for jd_id, payload in payloads["data_by_jd"].items():
        if not payload.startswith("Skills: ") or ". Experience: " not in payload or not payload.isascii():
            raise RuntimeError(f"Malformed data injection payload: {jd_id}")
    OUTPUT.mkdir(exist_ok=True)
    font_dir = OUTPUT / "tmp_fonts"
    font_dir.mkdir(exist_ok=True)
    overlays = {"instruction": overlay_bytes("instruction", payloads["instruction"], font_dir)}
    overlays.update({jd_id: overlay_bytes(jd_id, payload, font_dir)
                     for jd_id, payload in payloads["data_by_jd"].items()})
    records = []
    for index, pair in enumerate(pairs, 1):
        source_pdf = DATASET / pair["clean_pdf"]
        if hashlib.sha256(source_pdf.read_bytes()).hexdigest() != pair["clean_pdf_sha256"]:
            raise RuntimeError(f"Clean source PDF changed: {pair['pair_id']}")
        for kind in ("instruction", "data"):
            jd_id = pair["jd_id"]
            payload = payloads["instruction"] if kind == "instruction" else payloads["data_by_jd"][jd_id]
            overlay = overlays["instruction"] if kind == "instruction" else overlays[jd_id]
            filename = f"{pair['pair_id']}_{pair['source_id']}_{kind}.pdf"
            target = OUTPUT / kind / filename
            target.parent.mkdir(exist_ok=True)
            merge_resume(source_pdf, overlay, target)
            extracted, pages = check_pdf(source_pdf, target, payload)
            text_target = OUTPUT / f"{kind}_text" / filename.replace(".pdf", ".txt")
            text_target.parent.mkdir(exist_ok=True)
            text_target.write_text(extracted)
            records.append({
                "pair_id": pair["pair_id"], "jd_id": jd_id,
                "source_id": pair["source_id"], "fit_group": pair["fit_group"],
                "kind": kind,
                "clean_pdf": pair["clean_pdf"],
                "clean_pdf_sha256": pair["clean_pdf_sha256"],
                "injected_pdf": str(target.relative_to(DATASET)),
                "extracted_text": str(text_target.relative_to(DATASET)),
                "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "injected_pdf_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "source_pages": pages,
            })
        if index % 25 == 0:
            print(f"Validated injections for {index}/200 pairs", flush=True)
    if len(records) != 400:
        raise RuntimeError("Expected 200 instructional and 200 data injected PDFs")
    with (DATASET / "injected_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    shutil.rmtree(font_dir)
    print("Generated and validated 400 injected PDFs")


if __name__ == "__main__":
    main()
