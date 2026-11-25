# Orthogonal 2D Fourier-Centroid Bridging for Spatial Grounding in Endoscopic VQA

Code and result summaries for the IEEE Signal Processing Letters submission.

## Layout
- `src/` - `ortho_unified_core.py` (Stage 1 encoder + Stage 2 orthogonal bridge), `geofilm_bridge_core.py` (Additive Bridge baseline), `eval_energy_efficiency.py`, `extract_geope_real_fourier_vectors.py`
- `notebooks/` - training/evaluation notebooks 01, 04-08 (see numbering in the supplementary material)
- `results/` - small summary CSV/JSON files behind the paper tables

## Notes
- Notebooks were run on Google Colab and contain Drive paths (`/content/drive/MyDrive/...`). Edit the path cells before running.
- Large per-sample files (`predictions.json`, judge JSONs, checkpoints) are not included. [LINK_NEEDED: hosting location for large artifacts]
- Dataset: Kvasir-VQA-x1 (official test split); obtain it from its original source.
- Install: `pip install -r requirements.txt` (versions unpinned).

## License
[LICENSE_NEEDED: choose a license before making the repo public]

## Citation
[CITATION_NEEDED: add BibTeX after acceptance]
