"""Rank original Kaggle resume PDFs for the 20 archived job descriptions.

This is a lexical shortlist only. A later clean-PDF screen must verify fit;
the source category and rank are never treated as qualification labels.
"""

import csv
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "source_resumes_300" / "source_archive.zip"
OUTPUT = ROOT / "reconstructed_dataset_200"
JOBS = OUTPUT / "job_descriptions_20"
TOKEN = re.compile(r"[a-z][a-z0-9]+(?:[-.+#][a-z0-9]+)*")
STOP = set("""a about above after again against all am an and any are as at be
because been before being below between both but by can could did do does
doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more
most my myself no nor not now of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up
very was we were what when where which while who whom why will with you your
yours yourself yourselves job role team company candidate applicant ability
required experience work working strong good excellent including preferred
responsibilities qualifications position years year""".split())


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.casefold()) if t not in STOP and len(t) > 2]


def normalized_hash(text: str) -> str:
    value = " ".join(text.casefold().split())
    return hashlib.sha256(value.encode()).hexdigest()


def corpus() -> list[dict]:
    excluded = {r["source_id"] for r in csv.DictReader(
        (ROOT / "source_resumes_300" / "manifest.csv").open()
    )}
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive_names = set(archive.namelist())
        source_rows = list(csv.DictReader(io.TextIOWrapper(
            archive.open("Resume/Resume.csv"), encoding="utf-8-sig", errors="replace"
        )))
    seen_text = set()
    eligible = []
    for row in sorted(source_rows, key=lambda x: x["ID"]):
        source_id = row["ID"]
        source_path = f'data/data/{row["Category"]}/{source_id}.pdf'
        text = row["Resume_str"].strip()
        if source_id in excluded or source_path not in archive_names or len(text) < 500:
            continue
        text_hash = normalized_hash(text)
        if text_hash in seen_text:
            continue
        seen_text.add(text_hash)
        eligible.append({"source_id": source_id, "source_category": row["Category"],
                         "archive_entry": source_path, "source_text": text,
                         "source_text_sha256": text_hash})
    return eligible


def score_all(resumes: list[dict], query: str) -> list[float]:
    n = len(resumes)
    counts = [Counter(tokens(r["source_text"])) for r in resumes]
    df = Counter()
    for c in counts:
        df.update(c.keys())
    idf = {term: math.log((n + 1) / (frequency + 1)) + 1 for term, frequency in df.items()}
    q_counts = Counter(tokens(query))
    query_vector = {term: (1 + math.log(count)) * idf.get(term, math.log(n + 1))
                    for term, count in q_counts.items()}
    query_norm = math.sqrt(sum(v * v for v in query_vector.values())) or 1
    scores = []
    for c in counts:
        norm_sq = 0.0
        dot = 0.0
        for term, frequency in c.items():
            weight = (1 + math.log(frequency)) * idf[term]
            norm_sq += weight * weight
            dot += weight * query_vector.get(term, 0.0)
        scores.append(dot / ((math.sqrt(norm_sq) or 1) * query_norm))
    return scores


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    resumes = corpus()
    jobs = list(csv.DictReader((JOBS / "manifest.csv").open()))
    if len(jobs) != 20 or len(resumes) < 2000:
        raise RuntimeError("Expected 20 jobs and at least 2,000 new corpus PDFs")
    records = []
    for job in jobs:
        text = (JOBS / job["text_file"]).read_text()
        # Requirements are near the beginning of each saved posting. The
        # repeated title emphasizes the role over universal posting boilerplate.
        query = (job["title"] + " ") * 6 + text[:4500]
        scores = score_all(resumes, query)
        ordered = sorted(range(len(resumes)), key=lambda i: (-scores[i], resumes[i]["source_id"]))
        for rank, index in enumerate(ordered, start=1):
            resume = resumes[index]
            records.append({
                "jd_id": job["jd_id"], "rank": rank, "lexical_score": f"{scores[index]:.6f}",
                "source_id": resume["source_id"],
                "source_category": resume["source_category"],
                "archive_entry": resume["archive_entry"],
                "source_text_sha256": resume["source_text_sha256"],
            })
        print(f'{job["jd_id"]}: top {resumes[ordered[0]]["source_category"]} '
              f'{resumes[ordered[0]]["source_id"]} score={scores[ordered[0]]:.3f}', flush=True)
    with (OUTPUT / "rankings.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT / "ranking_method.json").write_text(json.dumps({
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "eligible_new_resume_count": len(resumes),
        "jobs": len(jobs),
        "excluded_old_sample": 300,
        "query": "title repeated six times plus first 4,500 characters of saved JD",
        "method": "log-TF-IDF cosine over Kaggle CSV text, selection only",
    }, indent=2) + "\n")
    print(f"Ranked {len(resumes)} new source PDFs against {len(jobs)} jobs")


if __name__ == "__main__":
    main()
