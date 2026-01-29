---
title: Challenge
---

# DGM⁴+ Challenge on Global Scene Inconsistency Detection


---

## Overview

The **DGM⁴+ Challenge on Global Scene Inconsistency Detection** aims to advance research on detecting and explaining **semantic inconsistencies** between visual content and accompanying text.

Modern multimodal misinformation increasingly relies on **contextual manipulation**, where fabricated foregrounds, misleading backgrounds, and deceptive captions are combined to produce persuasive but false narratives. Unlike traditional deepfake benchmarks that focus on low-level artifacts, **DGM⁴+ emphasizes scene-level and narrative-level inconsistency**.

This challenge provides a standardized benchmark for evaluating models that jointly reason over images and text to identify misleading content.

---

## Task Description

Given an **image–caption pair**, participants must jointly perform:

### 1) Authenticity Classification (Required)
Predict whether the pair is **semantically consistent (authentic)** or contains **contextual manipulation (manipulated)**.

### 2) Manipulation-Type Classification (Required)
For manipulated samples, predict whether the inconsistency corresponds to:

- **Text Swap**
- **Text Attribute**

### 3) Textual Grounding of Inconsistency (Required)
Identify caption **tokens or spans** that introduce misleading, fabricated, or contextually inconsistent information.

**All three outputs are mandatory and are jointly evaluated.**

---

## Dataset

The DGM⁴+ dataset contains approximately **5,000** news-style image–caption pairs, including:

- Authentic scenes  
- Text swap manipulations  
- Text attribute manipulations  
- Narrative reframing  

Dataset splits:

| Split | Availability | Purpose |
|------|--------------|---------|
| Training | Public | Model development |
| Validation | Public | Hyperparameter tuning |
| Test | Hidden | Leaderboard evaluation |

Public splits are released under a **CC BY-NC 4.0** license.

Dataset is available to view and download at:  
https://drive.google.com/file/d/1kXNqljyJ7EHmHnRn3ORPgLjawnJ63YHL/view?usp=sharing

For more information about the dataset:  
https://arxiv.org/abs/2509.26047

---

## Required Outputs

Each submission must provide predictions for **all test samples**.

### (a) Authenticity Prediction

A **CSV** file containing the following columns:

`id,label,confidence,manipulation_type`

Where:

- `label` ∈ {`authentic`, `manipulated`}  
- `confidence` ∈ [0, 1]  
- `manipulation_type` ∈ {`origin`, `text_swap`, `text_attribute`}  

For authentic samples, `manipulation_type` must be `origin`.

---

### (b) Text Grounding Output

A **JSON** file containing token indices corresponding to misleading or manipulated text.

Example:

    {
      "0001": [3,4,5,6]
    }

Tokenization follows the **official tokenizer** released with the dataset.

For authentic samples, an empty list must be submitted.

---

## Optional Cross-Modal Grounding and Explanation Track

In addition to required outputs, participants are encouraged to submit cross-modal explanations linking misleading textual elements to relevant visual evidence.

For each manipulated sample, teams may optionally provide:

- Alignments between identified misleading tokens and corresponding image regions, and/or  
- Visualizations illustrating how textual claims are supported or contradicted by visual content.  

These explanations are intended to demonstrate how models reason about semantic inconsistency across modalities.

Optional submissions are not included in the official leaderboard evaluation and do not affect ranking. However, high-quality cross-modal grounding and explanation outputs will be highlighted during the poster session and invited presentations.

Selected teams may receive a **Best Explainability Award**.

### Optional Submission Format

Optional explanations may be submitted as:

- Visualization images (PNG/JPEG), and/or  
- JSON files linking token indices to bounding boxes.

Example format:

    {
      "0001": {
        "tokens": [3,4,5],
        "regions": [[0.12,0.34,0.56,0.78]]
      }
    }

---

## Submission Format

Each submission must be packaged as:

    submission.zip
     ├── classification.csv
     └── text_grounding.json

---

## Evaluation

### 1) Classification Evaluation

Authenticity classification is evaluated using:

- Accuracy  
- F1-score  
- ROC–AUC  

Confidence scores are used for AUC computation.

### 2) Text Grounding and Type Evaluation

For manipulated samples, prediction of **text swap vs. text attribute** is evaluated using:

- Accuracy  
- Macro F1-score  

This evaluates the system’s ability to distinguish different forms of semantic inconsistency.

---

## Evaluation Protocol

Evaluation is conducted independently for each component and combined into a final score.

Final rankings are computed using:

Score = 0.5 × F1_binary  
      + 0.3 × F1_text  
      + 0.2 × F1_type

This weighting prioritizes reliable detection of semantic inconsistency while rewarding accurate explanation and manipulation-type recognition.

---

## Challenge Timeline

| Date | Milestone |
|------|-----------|
| Jan 10, 2026 | Challenge launch |
| Mar 05, 2026 | Submission portal opens |
| Mar 25, 2026 | Test set release |
| Apr 05, 2026 | Final submissions due |
| Apr 15, 2026 | Results announced |
| Jun 04, 2026 | On-site presentations and awards |

---

## Participation

To participate:

1. Register via the challenge platform (add link here)  
2. Download the training and validation sets  
3. Develop and evaluate models  
4. Submit predictions on the hidden test set  

Registration links and instructions will be provided on this page.

---

## Awards

Outstanding contributions will be recognized through:

- Best Overall Performance  
- Best Explainability and Grounding  
- Best Student Team  

Selected teams will be invited to present short papers and demos at PP-MisDet.

---

## Ethical Use Policy

All manipulated samples are synthetically generated and do not depict real individuals or events.

Participants agree to:

- Use the dataset exclusively for research purposes  
- Not generate or disseminate misinformation  
- Not redistribute the dataset without permission  

Violations may result in disqualification.

---

## Transparency and Reproducibility

After the challenge concludes, evaluation scripts, annotation guidelines, and benchmark statistics will be publicly released to support long-term community use.

---

## Updates

Additional information regarding baseline models, leaderboard access, and submission procedures will be posted on this page. Please check regularly for updates.
