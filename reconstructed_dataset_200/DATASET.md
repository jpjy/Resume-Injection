# Dataset card: reconstructed resume-screening benchmark

**Unit of analysis:** one archived job description paired with one original resume PDF. Twenty postings each have ten distinct source IDs; the same source can recur under a different posting. The 200 pairs use 138 distinct PDFs from `snehaanbhawal/resume-dataset` on Kaggle. For each pair, the saved comparison contains `clean`, `instruction`, and `data` conditions. The two attack conditions were generated from the clean PDF with a private-use-code font and divergent PDF `/ToUnicode` mapping.

**Selection:** an earlier GPT-4o screen selected five likely-fit and five likely-nonfit resumes per posting. These are model judgments, not human qualification labels. Reusing a source PDF across postings and selecting with one model can bias cross-model comparisons.

**Sources:** `manifest.csv` records each source archive entry and SHA-256, plus the job URL and archived text SHA-256. `job_descriptions_20/manifest.csv` gives each posting's title, company, URL, retrieval date, and text path. Original PDFs come from the public [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). The source archive and materialized PDFs are kept out of Git; `scripts/materialize_selected_pdfs.py` reconstructs the pair files and verifies byte hashes.

**Payloads:** `injection_payloads.json` contains one common screening command and one job-specific Skills/Experience text for each posting. Both are experimental attacker content. `injected_manifest.csv` records PDF and extracted-text paths and hashes for the 400 variants. Regeneration requires the exact font file described in `assets/README.md`.

**Evaluations:** `all_models_comparison_600.csv` has one row per pair and condition (600 rows). It includes `pair_id`, `jd_id`, `source_id`, `fit_group`, `condition`, and `pdf_file`, followed by each of four models' `score`, `pass_level`, `pass`, and one-sentence `reason`. Thus the file represents 2,400 model judgments. A score of at least 70 is a pass; the saved five-level rubric is not a real hiring policy.

**Limits:** the benchmark is a controlled screening simulation. We did not verify the resume examples' qualifications, any generated skill claim, actual employment decisions, human visual detection, OCR behavior, or repeated sampling. Archived postings can disappear or change online; use the saved text and hash for exact reproduction. The PDF discrepancy was checked with PyMuPDF and may behave differently in other extractors.
