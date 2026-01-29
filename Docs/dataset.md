---
title: Dataset Details
---

# Dataset Details (DGM⁴+)

DGM⁴+ extends DGM⁴-style synthetic generation by creating **foreground–background (FG–BG) mismatches** with optional text attribute (TA) and text swap (TS) manipulations. :contentReference[oaicite:3]{index=3}

## Manipulation types in DGM⁴+
DGM⁴+ introduces global inconsistency and hybrids:

- **FG–BG (origin/generated)**: global mismatch in the image background  
- **FG–BG + TA (manipulation/text_attribute)**: global mismatch + affect/attribute shift in text  
- **FG–BG + TS (manipulation/text_swap)**: global mismatch + irrelevant/swapped text span  

These correspond to the output directories described in this repository. :contentReference[oaicite:4]{index=4}

## Generation constraints (quality control)
The pipeline includes:
- face gating (e.g., enforce 1–3 faces)
- deduplication using pHash
- optional OCR-based text scrubbing
- DGM⁴-style resize/crop to 400×256 :contentReference[oaicite:5]{index=5}

## Intended use
DGM⁴+ is designed to stress-test multimodal misinformation detectors on **scene-level plausibility reasoning** rather than only local artifact detection.

## Citation
Add your BibTeX here (use the one from your paper).
