# Resume-Injection

Code and benchmark data for a paired study of PDF font injection in LLM-assisted resume screening. The reconstructed benchmark uses **20 archived job postings**, **200 resume–job pairs** (five baseline likely-fit and five likely-nonfit per posting), and **138 distinct original resume PDFs** from the [Snehaan Bhawal Kaggle archive](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset). Each pair has a clean, instruction-injected, and job-specific data-injected condition. The four-model comparison records **2,400 evaluations**, including a score, five-level recommendation, and one-sentence reason for each evaluation.

The likely-fit labels come from an earlier GPT-4o screen; they are not verified hiring outcomes. The added skills and experience are experimental attacker claims, not verified facts. This repository documents a defensive research benchmark and makes no real hiring decisions.

## Layout

| Path | Contents |
| --- | --- |
| `scripts/` | Posting collection, candidate ranking/selection, PDF construction, model evaluation, and comparison code |
| `source_resumes_300/manifest.csv` | Source archive mapping for the earlier sample excluded from the reconstructed benchmark |
| `reconstructed_dataset_200/manifest.csv` | The 200 selected pairs, Kaggle archive entries, source hashes, posting URLs/hashes, and baseline strata |
| `reconstructed_dataset_200/job_descriptions_20/` | The 20 archived postings and their source URL/hash manifest |
| `reconstructed_dataset_200/injection_payloads.json` | One job-independent instruction and 20 posting-specific skill/experience payloads |
| `reconstructed_dataset_200/injected_manifest.csv` | Paths and hashes for 400 injected PDF variants |
| `reconstructed_dataset_200/clean_pdfs/` | 200 browsable clean resume PDFs, organized by job-description ID |
| `reconstructed_dataset_200/font_injected_pdfs/instruction/` | 200 instruction-injected resume PDFs |
| `reconstructed_dataset_200/font_injected_pdfs/data/` | 200 job-specific data-injected resume PDFs |
| `reconstructed_dataset_200/resume_pdf_dataset_200.zip` | Downloadable archive of 200 clean and 400 injected PDFs, plus extracted attack text |
| `reconstructed_dataset_200/all_models_comparison_600.csv` | Paired clean/instruction/data results for GPT-4o, GPT-5.5, Claude Haiku 4.5, and Claude Opus 4.6 |

The 600 individual PDFs are available as browsable repository files and in the versioned ZIP for bulk download. The full original Kaggle archive and experiment font binary are not redistributed. The manifests, payloads, archived job texts, and model results are versioned. See [the dataset card](reconstructed_dataset_200/DATASET.md) for provenance and fields.

## Rebuild the PDF conditions

For direct inspection, unzip `reconstructed_dataset_200/resume_pdf_dataset_200.zip` inside `reconstructed_dataset_200/`. To independently rebuild, install the Python dependencies in `requirements.txt`, download the original Kaggle dataset, and save its ZIP as `source_resumes_300/source_archive.zip`. The code verifies source PDF SHA-256 hashes before materializing the 200 pair files:

```bash
python scripts/materialize_selected_pdfs.py
```

The injection generator also requires the exact Liberation Sans 1.07.4 font recorded in [`assets/README.md`](assets/README.md). Place it at `assets/LiberationSans-Regular.ttf`, then run:

```bash
python scripts/generate_reconstructed_injections.py
```

This creates 200 instruction and 200 job-specific data PDFs. It checks page count, extraction order, payload presence, and preservation of source text. The instruction is identical across postings; the data condition uses one distinct payload per posting.

`scripts/evaluate_reconstructed_gpt4o_all.py`, `evaluate_reconstructed_gpt55_all.py`, and `evaluate_reconstructed_claude_all.py` rerun API evaluations. They request keys from the environment or an interactive prompt; no key is stored in this repository. API calls are not needed to inspect the saved results.

## Publication notes

The Kaggle dataset describes the original resume examples as collected from LiveCareer. Source identifiers and hashes are supplied so readers can retrieve the public archive directly. The archived job descriptions include source URLs and retrieval dates. We have not assigned a new license to the third-party resume or posting content.
