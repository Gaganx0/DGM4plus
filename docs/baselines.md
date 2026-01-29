---
title: Baselines
---


# Baseline Models

---

To facilitate rapid onboarding, we recommend the following baselines.

---

## Vision–Language Similarity Baseline

### CLIP / SigLIP

Use cosine similarity between image and caption embeddings.

Purpose:
- Detect global semantic mismatch
- Provide fast initial benchmark

---

## Vision-Only Baseline

### DINOv2 Features

Compare foreground and background embeddings.

Purpose:
- Measure visual inconsistency
- Analyze scene composition

---

## Multimodal Reasoning Baseline

### Multimodal LLM (Qwen2-VL)

Prompt model with:

"Is this image–caption pair plausible?"

Purpose:
- Test reasoning ability
- Establish upper-bound reference

---

## Manipulation-Aware Baseline

### HAMMER

Evaluate local manipulation detection failure modes on FG–BG data.

Purpose:
- Study out-of-distribution behavior

---

## Reporting Guidelines

Baseline reports should include:

- Hardware used
- Inference time
- Training data
- Hyperparameters
- Failure cases
