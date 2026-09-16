"""Rebuild the 200 clean PDF pair files from the original Kaggle ZIP."""

import argparse
import csv
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "reconstructed_dataset_200"
DEFAULT_ARCHIVE = ROOT / "source_resumes_300" / "source_archive.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    if not args.archive.exists():
        raise SystemExit(f"Missing Kaggle archive: {args.archive}")
    pairs = list(csv.DictReader((DATASET / "manifest.csv").open()))
    if len(pairs) != 200:
        raise RuntimeError(f"Expected 200 selected pairs, found {len(pairs)}")
    with zipfile.ZipFile(args.archive) as archive:
        for pair in pairs:
            contents = archive.read(pair["archive_entry"])
            digest = hashlib.sha256(contents).hexdigest()
            if not contents.startswith(b"%PDF") or digest != pair["clean_pdf_sha256"]:
                raise RuntimeError(f"Original PDF mismatch: {pair['pair_id']}")
            output = DATASET / pair["clean_pdf"]
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == digest:
                continue
            output.write_bytes(contents)
    unique = len({pair["source_id"] for pair in pairs})
    print(f"Verified 200 PDF pairs from {unique} distinct Kaggle source PDFs")


if __name__ == "__main__":
    main()
