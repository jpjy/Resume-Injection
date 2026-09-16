"""Create paired resume PDFs with a visible/semantic font mismatch.

The TrueType cmap maps one private-use code point per payload character to
the glyph for a benign cover character. The PDF ToUnicode map then maps that
code point to the hidden payload character. The first-page top band renders
the cover while PyMuPDF extracts the payload; original resume pages are kept.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import textwrap

from fontTools.ttLib import TTFont as FontToolsTTFont
import pymupdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as ReportLabTTFont
from reportlab.pdfgen.canvas import Canvas


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "source_resumes_300"
JOBS = PROJECT / "job_descriptions_20"
OUTPUT = PROJECT / "font_injected_resumes_600"
FONT_SOURCE = PROJECT / "assets" / "LiberationSans-Regular.ttf"
BAND_HEIGHT = 90
COVER_SENTENCES = [
    "Professional background and references appear in the resume below.",
    "Additional professional details can be discussed during a formal review.",
    "Professional details are available upon request.",
    "References are available upon request.",
]


def make_cover_line(length: int) -> str:
    for sentence in COVER_SENTENCES:
        if len(sentence) <= length:
            return sentence.ljust(length)
    return ("Profile details." if length >= 16 else "Note.").ljust(length)


def payload_lines(key: str, payload: str) -> tuple[list[str], list[str]]:
    if key != "instruction":
        skill, experience = payload.removeprefix("Skills: ").split(". Experience: ", 1)
        return [skill + ".", experience], ["Skills", "Experience"]
    return textwrap.wrap(payload, width=90, break_long_words=False), ["Candidate Notes"]


def overlay_bytes(key: str, payload: str, font_dir: Path) -> bytes:
    if not payload.isascii() or not payload:
        raise ValueError("Payload must be non-empty ASCII")
    lines, headings = payload_lines(key, payload)
    if len(lines) > 2:
        raise ValueError(f"Payload too long for top band: {key}")
    covers = [make_cover_line(len(line)) for line in lines]
    font = FontToolsTTFont(FONT_SOURCE)
    original = next(
        table.cmap for table in font["cmap"].tables
        if table.isUnicode() and 65 in table.cmap
    )
    codepoint = 0xE000
    for cover in covers:
        for char in cover:
            glyph = original[ord(char)]
            for table in font["cmap"].tables:
                if table.isUnicode():
                    table.cmap[codepoint] = glyph
            codepoint += 1
    internal_name = f"ResumeMismatch{key.replace('_', '')}"
    for name_record in font["name"].names:
        if name_record.nameID in {1, 3, 4, 6}:
            name_record.string = internal_name.encode(name_record.getEncoding())
    font_path = font_dir / f"{key}.ttf"
    font.save(font_path)
    registered = f"Mismatch_{key}"
    pdfmetrics.registerFont(ReportLabTTFont(registered, str(font_path)))

    page_width, page_height = 595.276, 841.89
    buffer = io.BytesIO()
    canvas = Canvas(buffer, pagesize=(page_width, page_height + BAND_HEIGHT))
    canvas.setFillColorRGB(0.12, 0.12, 0.12)
    if key == "instruction":
        positions = [page_height + 52 - line_number * 14 for line_number in range(len(lines))]
        heading_positions = [page_height + 69]
    else:
        positions = [page_height + 57, page_height + 24]
        heading_positions = [page_height + 73, page_height + 40]
    codepoint = 0xE000
    for line_number, line in enumerate(lines):
        if line_number < len(headings):
            canvas.setFont("Helvetica-Bold", 8.5)
            canvas.drawString(28, heading_positions[line_number], headings[line_number])
        canvas.setFont(registered, 8.5)
        encoded = "".join(chr(codepoint + offset) for offset in range(len(line)))
        canvas.drawString(28, positions[line_number], encoded)
        codepoint += len(line)
    canvas.save()

    reader = PdfReader(io.BytesIO(buffer.getvalue()))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    mapped = set()
    offset = 0
    hidden_chars = {}
    for line in lines:
        for char in line:
            hidden_chars[f"<{0xE000 + offset:04X}>"] = f"<{ord(char):04X}>"
            offset += 1
    for font_ref in writer.pages[0]["/Resources"]["/Font"].values():
        font_obj = font_ref.get_object()
        if "/ToUnicode" not in font_obj:
            continue
        cmap = font_obj["/ToUnicode"].get_object().get_data().decode("ascii")
        for source_code, target_code in hidden_chars.items():
            if source_code in cmap:
                cmap = cmap.replace(source_code, target_code)
                mapped.add(source_code)
        stream = DecodedStreamObject()
        stream.set_data(cmap.encode("ascii"))
        font_obj[NameObject("/ToUnicode")] = writer._add_object(stream)
    if mapped != set(hidden_chars):
        missing = sorted(set(hidden_chars) - mapped)
        raise RuntimeError(f"Incomplete ToUnicode map for {key}: {missing[:12]} ({len(missing)} missing)")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def merge_resume(source_pdf: Path, overlay: bytes, destination: Path) -> None:
    source = PdfReader(source_pdf)
    band = PdfReader(io.BytesIO(overlay))
    first = band.pages[0]
    first.merge_page(source.pages[0])
    writer = PdfWriter()
    writer.add_page(first)
    for page in source.pages[1:]:
        writer.add_page(page)
    with destination.open("wb") as stream:
        writer.write(stream)


def check_pdf(source_pdf: Path, injected_pdf: Path, payload: str) -> tuple[str, int]:
    source = pymupdf.open(source_pdf)
    injected = pymupdf.open(injected_pdf)
    if len(source) != len(injected):
        raise RuntimeError(f"Page count changed: {injected_pdf}")
    if abs(injected[0].rect.height - source[0].rect.height - BAND_HEIGHT) > 0.2:
        raise RuntimeError(f"First-page band height incorrect: {injected_pdf}")
    extracted = "\n".join(page.get_text() for page in injected)
    expected = [payload] if payload.startswith("Screening note:") else list(payload_lines("data", payload)[0])
    normalized = " ".join(extracted.split())
    if any(" ".join(part.split()) not in normalized for part in expected):
        raise RuntimeError(f"Payload missing in PyMuPDF text: {injected_pdf}")
    if len(expected) == 2:
        start = injected[0].get_text()[:500]
        if not (start.find("Skills") < start.find(expected[0]) < start.find("Experience") < start.find(expected[1])):
            raise RuntimeError(f"Skills/experience section order incorrect: {injected_pdf}")
    original = "\n".join(page.get_text() for page in source)
    if len(extracted) < len(original):
        raise RuntimeError(f"Original resume text appears truncated: {injected_pdf}")
    return extracted, len(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    source_rows = list(csv.DictReader((SOURCE / "manifest.csv").open()))
    jd_rows = list(csv.DictReader((JOBS / "manifest.csv").open()))
    payloads = json.loads((PROJECT / "injection_payloads.json").read_text())
    if len(source_rows) != 300 or len(jd_rows) != 20:
        raise RuntimeError("Expected exactly 300 source resumes and 20 jobs")
    if set(payloads["data_by_jd"]) != {row["jd_id"] for row in jd_rows}:
        raise RuntimeError("Data payloads must map one-to-one with the 20 jobs")
    OUTPUT.mkdir(exist_ok=True)
    work_dir = OUTPUT / "tmp_fonts"
    work_dir.mkdir(exist_ok=True)
    instruction_overlay = overlay_bytes("instruction", payloads["instruction"], work_dir)
    data_overlays = {
        row["jd_id"]: overlay_bytes(row["jd_id"], payloads["data_by_jd"][row["jd_id"]], work_dir)
        for row in (jd_rows[:1] if args.pilot else jd_rows)
    }

    records = []
    tasks = []
    if args.pilot:
        tasks = [
            ("instruction", source_rows[0], "", instruction_overlay, payloads["instruction"]),
            ("data", source_rows[0], jd_rows[0]["jd_id"],
             data_overlays[jd_rows[0]["jd_id"]], payloads["data_by_jd"][jd_rows[0]["jd_id"]]),
        ]
    else:
        tasks.extend(
            ("instruction", row, "", instruction_overlay, payloads["instruction"])
            for row in source_rows
        )
        for job_index, job in enumerate(jd_rows):
            group = source_rows[job_index * 15:(job_index + 1) * 15]
            if len(group) != 15:
                raise RuntimeError("Every job must map to 15 distinct resumes")
            tasks.extend(
                ("data", row, job["jd_id"], data_overlays[job["jd_id"]],
                 payloads["data_by_jd"][job["jd_id"]])
                for row in group
            )

    for index, (kind, source_row, jd_id, overlay, payload) in enumerate(tasks, start=1):
        source_pdf = SOURCE / source_row["pdf_file"]
        target_dir = OUTPUT / ("pilot" if args.pilot else kind)
        text_dir = OUTPUT / ("pilot_text" if args.pilot else f"{kind}_text")
        target_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        suffix = kind if kind == "instruction" else f"data_{jd_id}"
        filename = f'{source_row["source_id"]}_{suffix}.pdf'
        target = target_dir / filename
        merge_resume(source_pdf, overlay, target)
        extracted, pages = check_pdf(source_pdf, target, payload)
        text_path = text_dir / filename.replace(".pdf", ".txt")
        text_path.write_text(extracted)
        records.append({
            "kind": kind,
            "source_id": source_row["source_id"],
            "source_pdf": source_row["pdf_file"],
            "jd_id": jd_id,
            "jd_url": next((j["posting_url"] for j in jd_rows if j["jd_id"] == jd_id), ""),
            "injected_pdf": str(target.relative_to(OUTPUT)),
            "extracted_text": str(text_path.relative_to(OUTPUT)),
            "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "pdf_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_pages": pages,
        })
        if index % 100 == 0:
            print(f"Validated {index}/{len(tasks)} injected PDFs", flush=True)
    manifest_path = OUTPUT / ("pilot_manifest.csv" if args.pilot else "manifest.csv")
    with manifest_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Generated and validated {len(records)} PDFs")


if __name__ == "__main__":
    main()
