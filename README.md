# Datasetgen.py

This repository provides the script used to generate a **DGM⁴-style synthetic dataset** with **absurd backdrops** and **news-style captions**.  
It extends HAMMER/DGM⁴ methodology by creating **foreground–background mismatches**, with optional **text attribute (TA)** and **text swap (TS)** manipulations.

---

## Repository Layout
```
DGM4plus/
├─ README.md
├─ Datasetgen.py   # Main dataset builder script
├─ worldcities.csv | cities1000.txt            # Required dateline file
├─ DGM4/                                       # Output dataset root
│  ├─ origin/generated/                        # Literal FG–BG
│  ├─ manipulation/text_attribute/             # FG–BG + TA
│  └─ manipulation/text_swap/                  # FG–BG + TS
└─ metadata.jsonl / metadata.json              # Incremental + snapshot metadata
```

---

## Quick start

### Python env
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate
pip install -U pip
pip install openai==1.* facenet-pytorch pillow imagehash tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # or your CUDA build
```

### OpenAI API key
```bash
export OPENAI_API_KEY=sk-...         # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
```

### Dateline file (required)
Place **one** of these next to the script, or set `CITY_DATA_PATH`:
- `worldcities.csv` (SimpleMaps)  
- `cities1000.txt` (GeoNames)

### (Optional) OCR hard mode
If you set `STRICT_NO_TEXT = True`, install Tesseract inside this env:
```bash
conda install -c conda-forge tesseract pytesseract
```

### Run
```bash
python Datasetgen.py
```

You’ll see a progress bar like:
```
Building 3000 DGM4-synthetic (seed=123456789)
```

---

## Key knobs (edit at top of the script)

- `OUT_ROOT = Path("DGM4")` — output directory.  
- `MISPLACED_BG_RATE = 1.0` — fraction of absurd backdrops (set `<1.0` to mix in realistic ones).  

**Bucket targets (total per run):**
- `TARGET_LITERAL` → FG–BG (stored under `origin/generated`)  
- `TARGET_INV_EMO` → FG–BG + TA (text attribute)  
- `TARGET_IRRELEV` → FG–BG + TS (text swap)  

**Concurrency:**  
- `CONCURRENCY = 5` — async workers hitting the image API.  

**Face gating & crop:**  
- `MIN_FACES=1`, `MAX_FACES=3`  
- `face_quality_gate(..., min_prob=0.80, min_side_px=110)`  
- Final size: `POST_W=400, POST_H=256` (DGM⁴-style)  

**OCR:**  
- `STRICT_NO_TEXT = False` (set `True` to blur detected text outside faces)  
- `TEXT_CONF_THRESH=60`, `OCR_SCALE=2`, `OCR_MAX_PASSES=1`  

**De-dup:**  
- pHash Hamming distance ≤ `PHASH_DISTANCE_MAX=3` is treated as duplicate  
- Caption strings are also de-duplicated  

**Safety/budget:**  
- `GEN_QUALITY="low"` for cost control  
- `MAX_IMAGE_CALLS=None` (set an integer to hard-limit API calls)  

**Resumability:**  
- Existing images & `metadata.jsonl/json` are scanned, and remaining remainder is generated.  
- Snapshots are written every `SNAP_EVERY=50`.  

---

## How captions are formed

- **Literal FG–BG** (`fake_cls="origin"`): neutral news-style headline with a normalized backdrop clause (`in/at/on/while …`).  
- **Text Attribute (TA)** (`"text_attribute"`): literal + affective add-on (e.g., “appears anxious”), with token spans in `fake_text_pos`.  
- **Text Swap (TS)** (`"text_swap"`): irrelevant subject/action/place headline; `fake_text_pos` marks the swapped span.  

*Grammatical helpers handle subject/verb agreement, punctuation, dateline punctuation vs em-dash, and headline tidying.*

---

## Prompts & backdrops

- **Absurd backdrops:** a large curated list drives surreal/global mismatches (e.g., *“a corridor of suspended stardust…”*).  
- If `MISPLACED_BG_RATE < 1.0`, realistic backdrops are injected but must semantically match role/event tags.  

---

## Under the hood

- **Image generation:** `openai.images.generate(model="gpt-image-1", size="1024x1024", quality="low")`  
- **Face gating:** MTCNN (CPU) → enforce face count & quality; limited regeneration attempts.  
- **Crop:** shot-aware crop (wide/medium/close-up) with union/center heuristics → resize to `400×256`.  
- **OCR scrub (optional):** upscale → detect → blur words outside face boxes → recheck.  
- **Dedup:** pHash (80-bit) + caption string set.  
- **Crash-safety:** every accepted sample appends to `metadata.jsonl`; periodic `metadata.json` snapshots; on exit, a final snapshot is written.  

---

## Tips & troubleshooting

- **Too many rejects:** lower face thresholds (`min_prob`, `min_side_px`), or increase `MAX_REGEN_ATTEMPTS`.  
- **Cost control:** reduce targets; set `MAX_IMAGE_CALLS`; keep `GEN_QUALITY="low"`.  
- **OCR errors:** leave `STRICT_NO_TEXT=False` or ensure Tesseract is available inside the env.  
- **Datelines missing:** ensure `worldcities.csv` or `cities1000.txt` is present (or set `CITY_DATA_PATH`).  
- **Resume later:** just re-run. The script recognizes existing images & metadata and builds only the remainder.  

---

## Minimal dependencies

If you don’t use OCR hard mode:
```
openai
facenet-pytorch
pillow
imagehash
tqdm
torch
torchvision
```

*(Install `pytesseract` + `tesseract` only if `STRICT_NO_TEXT=True`.)*

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{2025dgm4plus,
  title={DGM$^4$+: A Dataset Extension for Global Scene Inconsistency},
  booktitle={},
  year={2025}
}
```
