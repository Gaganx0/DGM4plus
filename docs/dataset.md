---
title: Dataset
---
<p align="center">
  <a href="index.md"><b>Home</b></a> |
  <a href="dataset.md"><b>Dataset</b></a> |
  <a href="challenge.md"><b>Challenge</b></a> |
  <a href="download.md"><b>Download</b></a> |
  <a href="baselines.md"><b>Baselines</b></a> |
  <a href="faq.md"><b>FAQ</b></a>
</p>

<hr>

# The DGM⁴+ Dataset

## Overview

DGM⁴+ is an extension of the DGM⁴ benchmark designed to support research on **global scene inconsistency** in multimodal misinformation.

Unlike traditional datasets that focus on local visual artifacts, DGM⁴+ emphasizes **semantic plausibility between foreground, background, and textual narratives**.

<p align="center">
  <img src="assets/teaser.png" width="85%">
</p>

*Figure 1: Examples of global scene inconsistency in DGM⁴+.*

---

## Global Scene Inconsistency

In DGM⁴+, manipulation is introduced at the **scene level**.

Typical inconsistencies include:

- Realistic people placed in physically impossible environments
- Foreground–background mismatches
- Implausible social or geopolitical contexts
- Contradictory visual–textual narratives

These cases are challenging for models relying only on local cues.

---

## Manipulation Taxonomy

<p align="center">
  <img src="assets/taxonomy.png" width="75%">
</p>

*Figure 2: Taxonomy of manipulations in DGM⁴+.*

DGM⁴+ introduces three primary categories:

### 1. FG–BG (Foreground–Background Mismatch)
Image-only manipulation where the foreground is inserted into an implausible background.

### 2. FG–BG + Text Attribute (TA)
Global scene inconsistency combined with sentiment, stance, or affect changes in captions.

### 3. FG–BG + Text Swap (TS)
Global mismatch combined with irrelevant or misleading caption replacement.

These extend the original DGM⁴ local manipulations.

---

## Dataset Composition

<p align="center">
  <img src="assets/examples_grid.png" width="90%">
</p>

*Figure 3: Representative samples across manipulation categories.*

### Size

- Total new samples: **~5,000**
- Expansion over DGM⁴: **+2.1%**

### Distribution

| Category        | Samples |
|-----------------|----------|
| FG–BG           | 2,000    |
| FG–BG + TA      | 1,500    |
| FG–BG + TS      | 1,500    |

### Captions

- News-style headlines
- Length: 10–25 tokens
- Human-readable and semantically rich

---

## Data Generation Pipeline

<p align="center">
  <img src="assets/pipeline.png" width="85%">
</p>

*Figure 4: Dataset generation pipeline.*

The dataset is generated using a multi-stage pipeline:

1. Foreground extraction
2. Background retrieval
3. Controlled composition
4. Caption rewriting
5. Quality filtering
6. Deduplication

---

## Quality Control

To ensure realism and diversity:

- Face-count constraints (1–3 faces)
- Perceptual hash deduplication
- OCR-based text removal
- Resolution normalization (400×256)

These steps reduce trivial detection shortcuts.

---

## Dataset Splits

| Split       | Availability | Purpose                  |
|-------------|--------------|--------------------------|
| Training    | Public       | Model development        |
| Validation  | Public       | Hyperparameter tuning    |
| Test        | Hidden       | Leaderboard evaluation   |

License: **CC BY-NC 4.0**

---

## Intended Use

DGM⁴+ is intended for:

- Multimodal misinformation detection
- Cross-modal reasoning
- Explainable AI research
- Robustness evaluation

---

## Citation

```bibtex
@article{singh2025dgm4plus,
  title   = {DGM$^4$+: Dataset Extension for Global Scene Inconsistency},
  author  = {Singh, Gagandeep and Amarsinghe, Samudi and Singh, Priyanka and Li, Xue},
  journal = {arXiv preprint arXiv:2509.26047},
  year    = {2025}
}
