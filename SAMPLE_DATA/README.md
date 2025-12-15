# SAMPLE_DATA

Tiny KO/EN dataset for out-of-the-box testing. All texts are synthetic/public-friendly.

## Files
- `docs_ko_en.jsonl`: mixed Korean/English short passages
- `metadata.json`: simple categories and sources

## Usage
- Ingestion scripts will read `SAMPLE_DATA/docs_ko_en.jsonl` and build a small ChromaDB.
- This is only for testing; replace with your own corpus for real deployments.

## Provenance
- Texts are paraphrased and non-sensitive, created for demo purposes.
