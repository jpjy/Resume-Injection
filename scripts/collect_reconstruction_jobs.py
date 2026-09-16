"""Snapshot 20 current Greenhouse posts suited to the source resume corpus."""

import csv
import hashlib
import html
import json
from pathlib import Path
from urllib.request import urlopen

from collect_job_descriptions import TextParser


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reconstructed_dataset_200" / "job_descriptions_20"
CACHE = Path("/private/tmp/greenhouse-boards")
JOBS = [
    ("lever:bespokepost", "556dedb6-cb03-40f8-a4b3-45c2d5e0fd66"),
    ("figma", 6119180004),
    ("figma", 6191672004), ("figma", 6119198004),
    ("stripe", 8180390),
    ("stripe", 6570253), ("stripe", 8165278),
    ("stripe", 7597613), ("stripe", 7617049),
    ("lever:bellotalabs", "a77e940d-1052-4363-802f-d64322a718fe"),
    ("robinhood", 8119781), ("robinhood", 8051290),
    ("lever:numeris", "fdc2c88a-cf4c-45f2-892e-0dbef1bcf5d0"),
    ("lever:ana-corp", "09422548-cf87-432b-8e08-1293de56274d"),
    ("stripe", 8159355),
    ("lever:promenade", "047dda0d-f2b7-4157-9a53-8b27c74e0750"),
    ("lever:promenade", "4d44c10a-9621-4e03-8e52-69ffbe633abb"),
    ("datadog", 7405831),
    ("lever:promenade", "baa8fb41-da6b-4c49-989a-9d7d26e275f6"),
    ("datadog", 8041182),
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    jobs_dir = OUTPUT / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    boards = {}
    for board in sorted({board for board, _ in JOBS if not board.startswith("lever:")}):
        cache_file = CACHE / f"{board}.json"
        if cache_file.exists():
            raw = cache_file.read_bytes()
        else:
            url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
            with urlopen(url, timeout=30) as response:
                raw = response.read()
        boards[board] = {job["id"]: job for job in json.loads(raw)["jobs"]}
    manifest = []
    for index, (board, job_id) in enumerate(JOBS, 1):
        if board.startswith("lever:"):
            site = board.split(":", 1)[1]
            cache_file = Path("/private/tmp/lever-reconstruction") / f"{site}_{job_id}.json"
            if cache_file.exists():
                job = json.loads(cache_file.read_text())
            else:
                url = f"https://api.lever.co/v0/postings/{site}/{job_id}?mode=json"
                with urlopen(url, timeout=30) as response:
                    job = json.load(response)
            parts = [job.get("descriptionPlain", "")]
            for section in job.get("lists", []):
                cleaner = TextParser()
                cleaner.feed(section.get("content", ""))
                parts.append(section.get("text", "") + "\n" + cleaner.text())
            parts.append(job.get("additionalPlain", ""))
            text = "\n".join(part for part in parts if part)
            jid = f"jd_{index:02d}"
            snapshot = {
                "jd_id": jid,
                "company": site.title(),
                "board": board,
                "lever_posting_id": job_id,
                "title": job["text"],
                "location": job.get("categories", {}).get("location"),
                "posting_url": job["hostedUrl"],
                "api_url": f"https://api.lever.co/v0/postings/{site}/{job_id}?mode=json",
                "first_published": job.get("createdAt"),
                "retrieved_date": "2026-09-15",
                "description_text": text,
                "raw_posting": job,
            }
            if len(text) < 1000:
                raise RuntimeError(f"Lever job description too short: {site}/{job_id}")
            (jobs_dir / f"{jid}.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
            (jobs_dir / f"{jid}.txt").write_text(text)
            manifest.append({
                "jd_id": jid, "company": snapshot["company"], "title": snapshot["title"],
                "posting_url": snapshot["posting_url"],
                "text_file": f"jobs/{jid}.txt", "json_file": f"jobs/{jid}.json",
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text_chars": len(text), "retrieved_date": snapshot["retrieved_date"],
            })
            continue
        job = boards[board].get(job_id)
        if job is None:
            raise RuntimeError(f"Job is not in current board snapshot: {board}/{job_id}")
        cleaner = TextParser()
        cleaner.feed(html.unescape(job["content"]))
        text = cleaner.text()
        if len(text) < 1000:
            raise RuntimeError(f"Job description too short: {board}/{job_id}")
        jid = f"jd_{index:02d}"
        snapshot = {
            "jd_id": jid,
            "company": job.get("company_name", board.title()),
            "board": board,
            "greenhouse_job_id": job_id,
            "title": job["title"],
            "location": job["location"]["name"],
            "posting_url": job["absolute_url"],
            "api_url": f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{job_id}",
            "first_published": job.get("first_published"),
            "updated_at": job.get("updated_at"),
            "retrieved_date": "2026-09-15",
            "description_text": text,
            "description_html": job["content"],
        }
        (jobs_dir / f"{jid}.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
        (jobs_dir / f"{jid}.txt").write_text(text)
        manifest.append({
            "jd_id": jid,
            "company": snapshot["company"],
            "title": snapshot["title"],
            "posting_url": snapshot["posting_url"],
            "text_file": f"jobs/{jid}.txt",
            "json_file": f"jobs/{jid}.json",
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "text_chars": len(text),
            "retrieved_date": snapshot["retrieved_date"],
        })
    with (OUTPUT / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Archived {len(manifest)} job descriptions in {OUTPUT}")


if __name__ == "__main__":
    main()
