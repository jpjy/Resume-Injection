"""Snapshot 20 live company job posts from Greenhouse's public Job Board API."""

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.request import urlopen


JOBS = [
    ("figma", 6131079004), ("figma", 6119180004),
    ("figma", 6191672004), ("figma", 6119198004),
    ("figma", 5691886004),
    ("stripe", 7532733), ("stripe", 8165278),
    ("stripe", 8080614), ("stripe", 7617049),
    ("stripe", 7867389),
    ("robinhood", 8119781), ("robinhood", 8051290),
    ("robinhood", 7490017), ("robinhood", 7263592),
    ("robinhood", 7997218),
    ("datadog", 7194969), ("datadog", 8122915),
    ("datadog", 7405831), ("datadog", 8075664),
    ("datadog", 7997846),
]


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "br", "div", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_data(self, data):
        self.parts.append(data)

    def text(self):
        return "\n".join(
            " ".join(line.split())
            for line in "".join(self.parts).splitlines()
            if line.strip()
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("job_descriptions_20"))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    jobs_dir = args.output / "jobs"
    jobs_dir.mkdir(exist_ok=True)
    boards = {}
    for board in sorted({board for board, _ in JOBS}):
        url = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
        if args.cache_dir:
            data = (args.cache_dir / f"{board}.json").read_bytes()
        else:
            with urlopen(url, timeout=30) as response:
                data = response.read()
        boards[board] = {job["id"]: job for job in json.loads(data)["jobs"]}

    manifest = []
    for index, (board, job_id) in enumerate(JOBS, start=1):
        if job_id not in boards[board]:
            raise RuntimeError(f"Job {board}/{job_id} is no longer live")
        job = boards[board][job_id]
        cleaner = TextParser()
        cleaner.feed(html.unescape(job["content"]))
        text = cleaner.text()
        if len(text) < 1000:
            raise RuntimeError(f"Job {board}/{job_id} has too little description text")
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
    with (args.output / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Archived {len(manifest)} current job descriptions")


if __name__ == "__main__":
    main()
