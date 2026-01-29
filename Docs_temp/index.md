---
title: DGM⁴+ Dataset (Challenge Benchmark)
---

# DGM⁴+ Challenge Dataset: Global Scene Inconsistency Detection

DGM⁴+ is an extension of DGM⁴ designed for **global scene inconsistency**: cases where a realistic foreground is placed in an implausible or absurd background, often paired with caption manipulations to mimic real multimodal misinformation. :contentReference[oaicite:1]{index=1}

## What makes DGM⁴+ different
Most detectors focus on **local artifacts** (e.g., face edits, local compositing cues). DGM⁴+ targets **scene-level plausibility** and **narrative consistency**: the foreground may look photorealistic, but the background context breaks semantics.

## Challenge task
Given an image–caption pair, participants predict whether it is **semantically consistent**.

### Optional (encouraged): grounding/explanations
Participants are encouraged to show:
- image regions that indicate inconsistency (often background evidence)
- caption spans that drive the decision  
This is **not scored**, but can be showcased in workshop presentations.

## Pages
- **[Dataset](dataset.md)**
- **[Download](download.md)**
- **[Evaluation](evaluation.md)**
- **[Baselines](baselines.md)**
- **[FAQ](faq.md)**

## Repository
This repo also includes the **generation pipeline** for DGM⁴+-style FG–BG mismatches and caption manipulations. :contentReference[oaicite:2]{index=2}

## Reference
Paper: https://arxiv.org/html/2509.26047v1
