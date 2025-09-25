# HAMMERAI_async_headlines_absurd_strong.py
# DGM4-style synthetic builder (extreme absurd backdrops + news-style captions)
# - Grammar/phrasing layer (plural agreement, punctuation, dateline/backdrop connectors)
# - Robust OCR scrub (scaled OCR, multiple passes, protects faces)
# - Face quality gating + limited regeneration
# - Em dash optional (USE_EM_DASH)
# - Budget: GEN_QUALITY kept "low"

import os, io, json, uuid, base64, random, re, asyncio, sys, csv
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

from PIL import Image, ImageFilter
import imagehash
from tqdm.auto import tqdm
from openai import AsyncOpenAI
from facenet_pytorch import MTCNN
from contextlib import suppress
import atexit, signal, time
from collections import Counter


# ---------------- CONFIG -----------------
OUT_ROOT = Path("DGM4")
IMG_MODEL  = "gpt-image-1"

# ---- Incremental metadata paths & cadence ----
META_JSONL = OUT_ROOT / "metadata.jsonl"   # append one JSON per line
META_SNAPSHOT = OUT_ROOT / "metadata.json" # rolling full snapshot
SNAP_EVERY = 50                            # write snapshot every N samples

# Image generation (we caption locally)
GEN_SIZE     = "1024x1024"
GEN_QUALITY  = "low"            # budget-friendly
POST_W, POST_H = 400, 256
JPEG_QUALITY = 85

MIN_FACES, MAX_FACES = 1, 3
PHASH_DISTANCE_MAX   = 3
CONCURRENCY          = 5

# % of absurd backdrops (0.0–1.0)
MISPLACED_BG_RATE = 1.0  # set <1.0 to mix in matched normal scenes

# Targets (common quick test: 5/5/5)
TARGET_LITERAL  = 0   # FG-BG
TARGET_INV_EMO  = 1500   # FG-BG + TA
TARGET_IRRELEV  = 1500  # FG-BG + TS
TOTAL_TARGET = TARGET_LITERAL + TARGET_INV_EMO + TARGET_IRRELEV
# Spend guard (None disables)
MAX_IMAGE_CALLS = None

# Random seed base (we’ll still add jitter)
SEED = None  # set to an int to make runs reproducible; None = fully random
if SEED is None:
    SEED = int.from_bytes(os.urandom(4), "little")
random.seed(SEED)

# OCR: enforce no legible text/logos
STRICT_NO_TEXT = False
TEXT_BLACKLIST = ["PRESS","HOSPITAL","FINISH","CENTER","POLICE","BANK","SCHOOL","STATION","UNIVERSITY","HOSPICE","GOVERNMENT"]
TEXT_CONF_THRESH = 60  # OCR confidence threshold (0-100)
OCR_SCALE = 2          # upscale factor before OCR for better detection
OCR_MAX_PASSES = 1     # blur-then-recheck passes

# Datelines: REQUIRED – no fallback
CITY_DATA_PATH = None  # autodetects if None; must exist (worldcities.csv or cities1000.txt)

# Typography preference for patterns
USE_EM_DASH = False  # set True to use an em dash between dateline and clause
MAX_REGEN_ATTEMPTS = 1
# DGM4-style dirs
DIR_LITERAL = OUT_ROOT / "origin" / "generated"
DIR_INV     = OUT_ROOT / "manipulation" / "text_attribute"
DIR_IRR     = OUT_ROOT / "manipulation" / "text_swap"
for d in [DIR_LITERAL, DIR_INV, DIR_IRR]:
    d.mkdir(parents=True, exist_ok=True)

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY not set.")

aclient = AsyncOpenAI()
mtcnn = MTCNN(keep_all=True, device="cpu")

async def _with_backoff(fn, *, retries=5, base=0.4, jitter=0.2):
    """
    Exponential backoff for transient errors.
    Tries 'retries' times; waits base*(2^i) + uniform(0,jitter) between tries.
    Pass in a zero-arg async callable 'fn'.
    """
    for i in range(retries):
        try:
            return await fn()
        except Exception:
            if i == retries - 1:
                raise
            await asyncio.sleep(base * (2 ** i) + random.uniform(0, jitter))


# ---------------- OCR -----------------
OCR_OK = False
if STRICT_NO_TEXT:
    try:
        import pytesseract
        cand = Path(sys.prefix) / "Library" / "bin" / "tesseract.exe"
        if cand.exists():
            pytesseract.pytesseract.tesseract_cmd = str(cand)
        _ = pytesseract.get_tesseract_version()
        OCR_OK = True
    except Exception:
        OCR_OK = False
    if not OCR_OK:
        raise SystemExit(
            "STRICT_NO_TEXT=True but Tesseract not available.\n"
            "Install in this env:  conda install -c conda-forge tesseract pytesseract"
        )

def _ocr_data(img: Image.Image):
    """Run OCR on a scaled copy for better detection."""
    import pytesseract
    if OCR_SCALE > 1:
        big = img.resize((img.width*OCR_SCALE, img.height*OCR_SCALE), Image.BICUBIC)
    else:
        big = img
    data = pytesseract.image_to_data(
        big, output_type=pytesseract.Output.DICT,
        config="--oem 1 --psm 6"  # LSTM, assume a block of text
    )
    return data

def has_readable_text(img: Image.Image) -> bool:
    try:
        data = _ocr_data(img)
        for i, txt in enumerate(data.get("text", []) or []):
            if not txt or len(txt) < 3: 
                continue
            try:
                conf = int(float(data["conf"][i]))
            except:
                conf = 0
            if conf >= TEXT_CONF_THRESH:
                return True
    except Exception:
        return False
    return False

def blur_rect(img, box, radius=6):
    x1,y1,x2,y2 = [max(0,int(v)) for v in box]
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2<=x1 or y2<=y1: return img
    region = img.crop((x1,y1,x2,y2)).filter(ImageFilter.GaussianBlur(radius))
    img.paste(region, (x1,y1))
    return img

def iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    iw=max(0, min(ax2,bx2)-max(ax1,bx1))
    ih=max(0, min(ay2,by2)-max(ay1,by1))
    inter=iw*ih
    if inter==0: return 0.0
    area=(ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/area if area>0 else 0.0

def scrub_text_if_detected(img: Image.Image, protected_boxes: List[List[int]] = None):
    """Blur high-confidence OCR words, avoiding face boxes; iterate and recheck."""
    import pytesseract
    protected_boxes = protected_boxes or []
    cur = img
    for _ in range(OCR_MAX_PASSES):
        dirty=False
        data = _ocr_data(cur)
        texts = data.get("text", []) or []
        for i, txt in enumerate(texts):
            if not txt or len(txt) < 3:
                continue
            try:
                conf = int(float(data["conf"][i]))
            except:
                conf = 0
            if conf >= TEXT_CONF_THRESH:
                # map scaled coords back
                x, y = int(data["left"][i]/OCR_SCALE), int(data["top"][i]/OCR_SCALE)
                w, h = int(data["width"][i]/OCR_SCALE), int(data["height"][i]/OCR_SCALE)
                box = (x-2,y-2,x+w+2,y+h+2)
                if any(iou(box, pb) > 0.15 for pb in protected_boxes):
                    continue
                cur = blur_rect(cur, box)
                dirty=True
        if not dirty:
            break
    # final check
    if has_readable_text(cur):
        return cur, True
    return cur, False

# ---------------- Scene vocabulary -----------------
ROLES = [
    "a politician","a spokesperson","a mayor","a health official","a coach","a referee","a scientist","a CEO",
    "a weather official","an activist","a news anchor","a reporter","a union representative","a judge","a fire chief",
    "a police spokesperson","a minister","a school principal","a researcher","a charity director","a local resident",
    "a parent","a teacher","a student","a nurse","a doctor","a shop owner","a barista","a delivery courier",
    "a bus driver","a train operator","a commuter","a farmer","a fisher","a factory worker","a mechanic",
    "a construction worker","an architect","an engineer","a small business owner","a volunteer","a librarian",
    "a museum curator","an art teacher","an athlete","a runner","a cyclist","a grocer","a baker","a chef","a waiter",
    "a bartender","a security guard","a police officer","a firefighter","an EMT","a paramedic","a postal worker",
    "a taxi driver","a rideshare driver","a pilot","a flight attendant","a sailor","a park ranger","a conservationist",
    "a professor","a school counselor","a PTA member","a voter","an election worker","a lawyer","a public defender",
    "a prosecutor","a witness","a city planner","an urban planner","a sanitation worker","a recycling worker",
    "a utility worker","a power line technician","a grid operator","a station manager","a ferry operator",
    "a dock worker","a warehouse worker","a hospital administrator","a nurse practitioner","a midwife",
    "a pharmacist","a lab technician","a veterinarian","an animal shelter worker","a homeowner","a neighbor",
    "a tourist","a hotel manager","a conference attendee","a trade show exhibitor","a community organizer",
    "a disaster relief worker"
]

EVENTS = [
    "addresses a press briefing","holds a town hall","meets neighborhood residents","visits a classroom",
    "opens a new library","visits a hospital ward","observes a vaccination clinic","thanks health workers",
    "visits a laboratory","announces research findings","reviews safety protocols","visits a fire station",
    "visits a police precinct","joins an emergency drill","inspects storm damage","distributes supplies",
    "inspects road repairs","announces transit upgrades","talks to commuters","walks through a factory",
    "meets union representatives","announces a jobs initiative","launches a pilot program","signs a partnership",
    "answers questions on the sidelines","meets a sports team","visits a community garden","plants a tree",
    "visits a recycling facility","discusses clean energy plans","visits a wind farm","visits a solar field",
    "meets housing advocates","cuts a ribbon at an opening","attends a vigil","visits a cultural center",
    "tours a museum exhibit","opens a book fair","visits an animal shelter","meets farmers at a fair",
    "visits a food bank","visits a tech lab","tests a prototype device","greets graduates","meets researchers",
    "answers questions in a corridor","walks with aides to a car","boards a commuter train"
]

# Normal (non-absurd) backdrops in case MISPLACED_BG_RATE < 1.0
BACKDROPS = [
    "on courthouse steps","inside a courtroom","in a media briefing room","inside a town hall chamber",
    "in a community center hall","on a neighborhood street","in a bright classroom","inside a public library",
    "in a hospital corridor","inside a research laboratory","in a fire station garage","inside a police precinct lobby",
    "on a factory floor","beside an assembly line","in a startup co-working space","in a modern boardroom",
    "on a sports sideline","in a locker room interview area","at a ballpark dugout rail"
]

# ------------- EXTREME ABSURD BACKDROPS (≈250) -------------
ABSURD_BACKDROPS = [
    "ice crystals ringing like tiny bells, as trains arrive without tracks",
    "an overpass braided from living vines, while lanterns orbit like quiet planets",
    "bridges curling like sleeping dragons, as doorways drift several inches off the ground",
    "at the edge of a square where footsteps leave glowing trails",
    "cliffs wrapped in slow, white fog ribbons, while benches slide gently along the pavement",
    "beside a field of silent, hovering moons",
    "a coastline under three pale moons, while streets rearrange their names",
    "streetlights blooming into glass flowers, while footprints glow faintly behind them",
    "beneath a bridge made of book spines",
    "a canyon of blue salt pillars, while fountains pour upward",
    "between a stadium whose seats drift like lily pads",
    "an avenue lined with sleeping giants, while birds trace geometric loops overhead",
    "wind carving letters into sand, as reflections move before the people do",
    "above a theatre whose red curtains pour like water",
    "underneath a shrine surrounded by levitating offerings",
    "stone heads emerging from dew-wet grass, as staircases knit across the air",
    "rain crystallizing into tiny prisms mid-air, while statues slowly turn their heads",
    "between a square where footsteps leave glowing trails",
    "a causeway guarded by silent, winged lions, as trains arrive without tracks",
    "beyond a museum with paintings that exhale fog",
    "beyond a library with shelves sprouting roots",
    "through a horizon where two suns trade places",
    "buildings folding like paper cranes, as shadows detach and wander",
    "inside a clocktower shedding gears like petals",
    "within the rings of a distant planet shimmering overhead",
    "against a bridge made of book spines",
    "beneath alleyways that narrow into pinpoints of light",
    "a desert where shadows point in different directions, while lanterns orbit like quiet planets",
    "underneath streets bending upward like ribbons",
    "a sky rumbling like a distant choir, while birds trace geometric loops overhead",
    "skyscrapers knotting together overhead, as kites hover without strings",
    "against a lake whose surface shows yesterday",
    "the rings of a distant planet shimmering overhead, while fountains pour upward",
    "a mausoleum that breathes mist, as distant mountains fold like accordions",
    "beneath a forest of crystal trunks resonating softly",
    "gusts that rearrange footprints, while lanterns orbit like quiet planets",
    "at the edge of a bridge that arcs into space and back",
    "beside snow that falls in slow spirals of ash-gray",
    "a shrine surrounded by levitating offerings, as trains arrive without tracks",
    "under a mausoleum that breathes mist",
    "underneath a square where footsteps leave glowing trails",
    "an overpass braided from living vines, as trains arrive without tracks",
    "across a stadium whose seats drift like lily pads",
    "within a stadium whose seats drift like lily pads",
    "row houses stitched together by laundry that never dries, while the moon appears in every window",
    "ice crystals ringing like tiny bells, while fountains pour upward",
    "along a steppe dotted with levitating boulders",
    "a terminal where platforms slowly slide past each other, as staircases knit across the air",
    "across a lighthouse sweeping beams that paint colors",
    "a plain where stones drift like soap bubbles, as reflections move before the people do",
    "an avenue lined with sleeping giants, while footprints glow faintly behind them",
    "on a parking garage filled with shallow, silver water",
    "an avenue lined with sleeping giants, as trains arrive without tracks",
    "against a horizon banded in impossible colors",
    "underneath ice crystals ringing like tiny bells",
    "a clocktower shedding gears like petals, while the moon appears in every window",
    "fields of wheat that turn to birds on the wind, while lanterns orbit like quiet planets",
    "underneath a waterfall that rises instead of falls",
    "a sky woven with unfamiliar constellations, while birds trace geometric loops overhead",
    "across a meadow carpeted with bioluminescent moss",
    "a sky tiled like a giant mosaic of galaxies, as distant mountains fold like accordions",
    "between a mangrove whose roots glow faintly",
    "against a reef of porcelain corals",
    "a cove where tides run on clockwork, while footprints glow faintly behind them",
    "above a valley where stars fall upward",
    "a sky rumbling like a distant choir, while streets rearrange their names",
    "a stadium whose seats drift like lily pads, as staircases knit across the air",
    "bridges curling like sleeping dragons, while birds trace geometric loops overhead",
    "a reef of porcelain corals, while birds trace geometric loops overhead",
    "underneath a desert where shadows point in different directions",
    "on a boulevard of doors that open to different skies",
    "a shrine surrounded by levitating offerings, as reflections move before the people do",
    "a squall of luminous pollen, while statues slowly turn their heads",
    "amid mirrored corridors reflecting into infinity",
    "a temple whose columns braid mid-air, as kites hover without strings",
    "through a sky tiled like a giant mosaic of galaxies",
    "underneath a cove where tides run on clockwork",
    "a clocktower shedding gears like petals, as trains arrive without tracks",
    "a boulevard of doors that open to different skies, while streets rearrange their names",
    "a glade where footprints sprout flowers, as trains arrive without tracks",
    "underneath a canyon of blue salt pillars",
    "at the edge of a sky rumbling like a distant choir",
    "a museum with paintings that exhale fog, as flags ripple without wind",
    "a hillside stitched with metallic vines, as staircases knit across the air",
    "a city sky filled with drifting auroras shaped like faces, while benches slide gently along the pavement",
    "inside stone heads emerging from dew-wet grass",
    "through a meadow carpeted with bioluminescent moss",
    "amid a canyon carved by time frozen mid-stream",
    "beside a cloister paved with moving constellations",
    "a causeway guarded by silent, winged lions, while the moon appears in every window",
    "between a lake whose surface shows yesterday",
    "a waterfall that rises instead of falls, while benches slide gently along the pavement",
    "a horizon where two suns trade places, as distant mountains fold like accordions",
    "inside drizzle that paints temporary murals",
    "a mangrove whose roots glow faintly, as trains arrive without tracks",
    "a theatre whose red curtains pour like water, while the moon appears in every window",
    "a square where rain falls upward as mist sinks, while birds trace geometric loops overhead",
    "between a colossus statue cracking open to reveal light",
    "a squall of luminous pollen, as flags ripple without wind",
    "sand dunes shaped like sleeping colossi, while fountains pour upward",
    "a bridge that arcs into space and back, as flags ripple without wind",
    "a sundial casting multiple times at once, while footprints glow faintly behind them",
    "a causeway guarded by silent, winged lions, as reflections move before the people do",
    "a gate held up by singing chains, while birds trace geometric loops overhead",
    "within a lighthouse sweeping beams that paint colors",
    "against row houses stitched together by laundry that never dries",
    "a lighthouse sweeping beams that paint colors, as trains arrive without tracks",
    "underneath rain crystallizing into tiny prisms mid-air",
    "underneath a valley where stars fall upward",
    "a marketplace where sound moves slower than light, while footprints glow faintly behind them",
    "across a bridge made of book spines",
    "a corridor of suspended stardust, as staircases knit across the air",
    "above pillars that drift like kites",
    "along a gate held up by singing chains",
    "through a causeway guarded by silent, winged lions",
    "mirrored corridors reflecting into infinity, as staircases knit across the air",
    "above a sundial casting multiple times at once",
    "beneath an amphitheatre that whispers names",
    "along a canyon of blue salt pillars",
    "across a square where footsteps leave glowing trails",
    "stone heads emerging from dew-wet grass, while birds trace geometric loops overhead",
    "a forest where gravity pulls sideways, while statues slowly turn their heads",
    "stone heads emerging from dew-wet grass, while lanterns orbit like quiet planets",
    "a garden where every leaf casts a different season of shadow, while benches slide gently along the pavement",
    "gusts that rearrange footprints, as shadows detach and wander",
    "a glade where footprints sprout flowers, while streets rearrange their names",
    "underneath a plaza where daylight and midnight meet in a line",
    "auroral curtains that trace silhouettes, while footprints glow faintly behind them",
    "an amphitheatre that whispers names, while lanterns orbit like quiet planets",
    "a garden where every leaf casts a different season of shadow, while the moon appears in every window",
    "a city sky filled with drifting auroras shaped like faces, as reflections move before the people do",
    "a canyon of blue salt pillars, while footprints glow faintly behind them",
    "a corridor of suspended stardust, while lanterns orbit like quiet planets",
    "a hallway without ends, only corners, as flags ripple without wind",
    "underneath a forest where gravity pulls sideways",
    "an arcade where prizes whisper forecasts, while statues slowly turn their heads",
    "a gate held up by singing chains, while lanterns orbit like quiet planets",
    "a square where footsteps leave glowing trails, as shadows detach and wander",
    "above a lake whose surface shows yesterday",
    "a waterfall that rises instead of falls, as staircases knit across the air",
    "a steppe dotted with levitating boulders, while statues slowly turn their heads",
    "mirrored corridors reflecting into infinity, as doorways drift several inches off the ground",
    "a sundial casting multiple times at once, as reflections move before the people do",
    "along a forest where gravity pulls sideways",
    "across a halo sun circled by faint duplicates",
    "under a canyon carved by time frozen mid-stream",
    "a marketplace where sound moves slower than light, as shadows detach and wander",
    "a field of silent, hovering moons, while the moon appears in every window",
    "beneath a staircase made of moving shadows",
    "a halo sun circled by faint duplicates, while fountains pour upward",
    "a city square paved with shifting clock faces, as reflections move before the people do",
    "an amphitheatre that whispers names, as doorways drift several inches off the ground",
    "between clouds patterned like fingerprints",
    "streets bending upward like ribbons, while benches slide gently along the pavement",
    "the rings of a distant planet shimmering overhead, while lanterns orbit like quiet planets",
    "an overpass braided from living vines, while footprints glow faintly behind them",
    "on a lake whose surface shows yesterday",
    "through cliffs wrapped in slow, white fog ribbons",
    "above a field of silent, hovering moons",
    "against sand dunes shaped like sleeping colossi",
    "alleyways that narrow into pinpoints of light, as doorways drift several inches off the ground",
    "above drizzle that paints temporary murals",
    "the rings of a distant planet shimmering overhead, as doorways drift several inches off the ground",
    "sand dunes shaped like sleeping colossi, while streets rearrange their names",
    "under a sky rumbling like a distant choir",
    "a bridge made of book spines, while lanterns orbit like quiet planets",
    "a theatre whose red curtains pour like water, as kites hover without strings",
    "within a canyon carved by time frozen mid-stream",
    "beside a library with shelves sprouting roots",
    "at the edge of a stadium whose seats drift like lily pads",
    "a pier extending into a lake of rippling glass, while birds trace geometric loops overhead",
    "pillars that drift like kites, while footprints glow faintly behind them",
    "a horizon where two suns trade places, while benches slide gently along the pavement",
    "beyond mirrored corridors reflecting into infinity",
    "on stone heads emerging from dew-wet grass",
    "a city square paved with shifting clock faces, while lanterns orbit like quiet planets",
    "a canyon carved by time frozen mid-stream, while streets rearrange their names",
    "between gusts that rearrange footprints",
    "across a sky rumbling like a distant choir",
    "a terminal where platforms slowly slide past each other, as shadows detach and wander",
    "mist that pools like liquid silver, as reflections move before the people do",
    "subway tunnels filled with quiet, drifting balloons, while birds trace geometric loops overhead",
    "gusts that rearrange footprints, as flags ripple without wind",
    "against pillars that drift like kites",
    "a sky tiled like a giant mosaic of galaxies, as reflections move before the people do",
    "a mausoleum that breathes mist, while lanterns orbit like quiet planets",
    "along a slow eclipse that never completes",
    "amid a squall of luminous pollen",
    "along an Escher-like staircase that loops into itself",
    "on drizzle that paints temporary murals",
    "a plain where stones drift like soap bubbles, as kites hover without strings",
    "streets bending upward like ribbons, as kites hover without strings",
    "pillars that drift like kites, while birds trace geometric loops overhead",
    "sand dunes shaped like sleeping colossi, while the moon appears in every window",
    "across a colossus statue cracking open to reveal light",
    "a sky stitched with ribbons of amber vapor, as distant mountains fold like accordions",
    "ice crystals ringing like tiny bells, as doorways drift several inches off the ground",
    "streets bending upward like ribbons, while fountains pour upward",
    "a city sky filled with drifting auroras shaped like faces, as flags ripple without wind",
    "under drizzle that paints temporary murals",
    "a theatre whose red curtains pour like water, as flags ripple without wind",
    "a parking garage filled with shallow, silver water, as kites hover without strings",
    "against a glade where footprints sprout flowers",
    "underneath cliffs wrapped in slow, white fog ribbons",
    "on a mangrove whose roots glow faintly",
    "a square where footsteps leave glowing trails, as doorways drift several inches off the ground",
    "a garden where every leaf casts a different season of shadow, as staircases knit across the air",
    "an archway opening to a pocket of night at noon, as shadows detach and wander",
    "between a horizon where two suns trade places",
    "an avenue lined with sleeping giants, as doorways drift several inches off the ground",
    "on a desert where shadows point in different directions",
    "under a squall of luminous pollen",
    "within a pier extending into a lake of rippling glass",
    "underneath a halo sun circled by faint duplicates",
    "a forest where gravity pulls sideways, as staircases knit across the air",
    "a bridge that arcs into space and back, while benches slide gently along the pavement",
    "on a moonlit sea reflecting a second, inverted skyline",
    "on rain crystallizing into tiny prisms mid-air",
    "above a sky stitched with ribbons of amber vapor",
    "skyscrapers knotting together overhead, as doorways drift several inches off the ground",
    "a valley where stars fall upward, as doorways drift several inches off the ground",
    "underneath wind carving letters into sand",
    "within bridges curling like sleeping dragons",
    "a pier extending into a lake of rippling glass, as kites hover without strings",
    "an avenue lined with sleeping giants, as distant mountains fold like accordions",
    "at the edge of a staircase made of moving shadows",
    "amid a mangrove whose roots glow faintly",
    "beyond a plain where stones drift like soap bubbles",
    "through an avenue lined with sleeping giants",
    "mist that pools like liquid silver, as shadows detach and wander",
    "above a desert where shadows point in different directions",
    "a library with shelves sprouting roots, as trains arrive without tracks",
    "through a halo sun circled by faint duplicates",
    "buildings folding like paper cranes, as kites hover without strings",
    "across stone heads emerging from dew-wet grass",
    "a garden where every leaf casts a different season of shadow, while fountains pour upward",
    "a mausoleum that breathes mist, while fountains pour upward",
    "on mist that pools like liquid silver",
    "a neighborhood tiled in repeating copies of itself, as flags ripple without wind",
    "along alleyways that narrow into pinpoints of light",
    "clouds patterned like fingerprints, as reflections move before the people do",
    "along a field of silent, hovering moons",
    "beneath subway tunnels filled with quiet, drifting balloons",
    "a moonlit sea reflecting a second, inverted skyline, as distant mountains fold like accordions",
    "beneath mirrored corridors reflecting into infinity",
    "a ziggurat whose staircases slide, while benches slide gently along the pavement",
    "a terminal where platforms slowly slide past each other, while streets rearrange their names",
    "through a slow eclipse that never completes",
    "beyond an Escher-like staircase that loops into itself",
    "beside clouds patterned like fingerprints",
    "beyond a causeway guarded by silent, winged lions",
    "under a forest where gravity pulls sideways",
    "on a reef of porcelain corals",
    "a mausoleum that breathes mist, as kites hover without strings",
    "a city square paved with shifting clock faces, as staircases knit across the air",
    "against a forest of crystal trunks resonating softly",
    "a valley where stars fall upward, while birds trace geometric loops overhead",
    "a forest where gravity pulls sideways, while benches slide gently along the pavement",
    "a bridge that arcs into space and back, as doorways drift several inches off the ground",
    "a sky woven with unfamiliar constellations, as reflections move before the people do",
    "a reef of porcelain corals, while footprints glow faintly behind them",
    "drizzle that paints temporary murals, while footprints glow faintly behind them",
    "a boulevard of doors that open to different skies, while footprints glow faintly behind them",
    "a halo sun circled by faint duplicates, as reflections move before the people do",
    "a library with shelves sprouting roots, as kites hover without strings",
    "a library with shelves sprouting roots, as staircases knit across the air",
    "amid fields of wheat that turn to birds on the wind",
    "sand dunes shaped like sleeping colossi, while footprints glow faintly behind them",
    "a sky woven with unfamiliar constellations, as shadows detach and wander",
    "across an overpass braided from living vines",
    "through a mangrove whose roots glow faintly",
    "a causeway guarded by silent, winged lions, as distant mountains fold like accordions",
    "a cloister paved with moving constellations, while lanterns orbit like quiet planets",
    "a pier extending into a lake of rippling glass, as reflections move before the people do",
    "between streetlights blooming into glass flowers",
    "ruins hovering one brick above the ground, as shadows detach and wander",
    "a moonlit sea reflecting a second, inverted skyline, as kites hover without strings",
    "through a sundial casting multiple times at once",
    "a valley where stars fall upward, while fountains pour upward",
    "a river hanging in the air, unmoving, while streets rearrange their names",
    "fields of wheat that turn to birds on the wind, as reflections move before the people do",
    "under a boulevard of doors that open to different skies",
    "the rings of a distant planet shimmering overhead, as kites hover without strings",
    "a ziggurat whose staircases slide, while statues slowly turn their heads",
    "a marketplace where sound moves slower than light, as doorways drift several inches off the ground",
    "a bridge made of book spines, as distant mountains fold like accordions",
    "a slow eclipse that never completes, while footprints glow faintly behind them",
    "amid a square where rain falls upward as mist sinks",
    "amid a slow eclipse that never completes",
    "underneath skyscrapers knotting together overhead",
    "a square where rain falls upward as mist sinks, while lanterns orbit like quiet planets",
    "underneath an overpass braided from living vines",
    "inside a city sky filled with drifting auroras shaped like faces",
    "on a horizon banded in impossible colors",
    "amid auroral curtains that trace silhouettes",
    "through a plain where stones drift like soap bubbles",
    "at the edge of a slow eclipse that never completes",
    "the rings of a distant planet shimmering overhead, while footprints glow faintly behind them",
    "ruins hovering one brick above the ground, as kites hover without strings",
    "underneath a canyon carved by time frozen mid-stream",
    "ice crystals ringing like tiny bells, while streets rearrange their names",
    "amid skyscrapers knotting together overhead",
    "snow that falls in slow spirals of ash-gray, while the moon appears in every window",
    "a library with shelves sprouting roots, while birds trace geometric loops overhead",
    "on a garden where every leaf casts a different season of shadow",
    "a forest of crystal trunks resonating softly, while statues slowly turn their heads",
    "underneath an avenue lined with sleeping giants",
    "above a cloister paved with moving constellations",
    "at the edge of a clocktower shedding gears like petals",
    "along a valley where stars fall upward",
    "amid a meadow carpeted with bioluminescent moss",
    "under a clocktower shedding gears like petals",
    "beside drizzle that paints temporary murals",
    "within a forest where gravity pulls sideways",
    "dawn light flickering like candle flames, as shadows detach and wander",
    "along streetlights blooming into glass flowers",
    "subway tunnels filled with quiet, drifting balloons, while the moon appears in every window",
    "within a valley where stars fall upward",
    "a mausoleum that breathes mist, as reflections move before the people do",
    "a hallway without ends, only corners, as distant mountains fold like accordions",
    "inside a temple whose columns braid mid-air",
    "a slow eclipse that never completes, as doorways drift several inches off the ground",
    "a gate held up by singing chains, while streets rearrange their names",
    "across a waterfall that rises instead of falls",
    "a plaza where daylight and midnight meet in a line, while streets rearrange their names",
    "on a bridge made of book spines",
    "a stadium whose seats drift like lily pads, as flags ripple without wind",
    "a square where footsteps leave glowing trails, as staircases knit across the air",
    "a city square paved with shifting clock faces, while streets rearrange their names",
    "on a steppe dotted with levitating boulders",
    "a temple whose columns braid mid-air, as reflections move before the people do",
    "on a square where footsteps leave glowing trails",
    "a canyon carved by time frozen mid-stream, as shadows detach and wander",
    "a canyon of blue salt pillars, while benches slide gently along the pavement",
    "along a library with shelves sprouting roots",
    "between a plaza where daylight and midnight meet in a line",
    "fields of wheat that turn to birds on the wind, as staircases knit across the air",
    "a slow eclipse that never completes, while lanterns orbit like quiet planets",
    "streetlights blooming into glass flowers, while the moon appears in every window",
    "a sky rumbling like a distant choir, as flags ripple without wind",
    "a colossus statue cracking open to reveal light, as staircases knit across the air",
    "skyscrapers knotting together overhead, while footprints glow faintly behind them",
    "above fog that parts into hallways",
    "a temple whose columns braid mid-air, while the moon appears in every window",
    "inside subway tunnels filled with quiet, drifting balloons",
    "within an arcade where prizes whisper forecasts",
    "within a glade where footprints sprout flowers",
    "on a canyon of blue salt pillars",
    "between a field of silent, hovering moons",
    "a square where footsteps leave glowing trails, as flags ripple without wind",
    "through a mausoleum that breathes mist",
    "an arcade where prizes whisper forecasts, while footprints glow faintly behind them",
    "above a museum with paintings that exhale fog",
    "an arcade where prizes whisper forecasts, as reflections move before the people do",
    "a meadow carpeted with bioluminescent moss, as staircases knit across the air",
    "a reef of porcelain corals, while fountains pour upward",
    "across a marketplace where sound moves slower than light",
    "bridges curling like sleeping dragons, while fountains pour upward",
    "beyond a lighthouse sweeping beams that paint colors",
    "beneath streets bending upward like ribbons",
    "on clouds patterned like fingerprints",
    "a meadow carpeted with bioluminescent moss, while birds trace geometric loops overhead",
    "against a marketplace where sound moves slower than light",
    "a neighborhood tiled in repeating copies of itself, as reflections move before the people do",
    "a sky stitched with ribbons of amber vapor, as shadows detach and wander",
    "against a museum with paintings that exhale fog",
    "above an overpass braided from living vines",
    "a valley where stars fall upward, while the moon appears in every window",
    "a boulevard of doors that open to different skies, while the moon appears in every window",
    "along a moonlit sea reflecting a second, inverted skyline",
    "a reef of porcelain corals, as reflections move before the people do",
    "fields of wheat that turn to birds on the wind, as distant mountains fold like accordions",
    "mist that pools like liquid silver, while benches slide gently along the pavement",
    "inside an avenue lined with sleeping giants",
    "under a museum with paintings that exhale fog",
    "a mausoleum that breathes mist, while the moon appears in every window",
    "a causeway guarded by silent, winged lions, while fountains pour upward",
    "the rings of a distant planet shimmering overhead, while benches slide gently along the pavement",
    "across a sky stitched with ribbons of amber vapor",
    "rain crystallizing into tiny prisms mid-air, while the moon appears in every window",
    "a plain where stones drift like soap bubbles, while streets rearrange their names",
    "beyond a desert where shadows point in different directions",
    "a coastline under three pale moons, while statues slowly turn their heads",
    "under a lighthouse sweeping beams that paint colors",
    "beside subway tunnels filled with quiet, drifting balloons",
    "beside a bridge made of book spines",
    "through a desert where shadows point in different directions",
    "beneath a sky woven with unfamiliar constellations",
    "a horizon banded in impossible colors, while lanterns orbit like quiet planets",
    "amid a library with shelves sprouting roots",
    "at the edge of a desert where shadows point in different directions",
    "sand dunes shaped like sleeping colossi, as shadows detach and wander",
    "a colossus statue cracking open to reveal light, as flags ripple without wind",
    "amid a cove where tides run on clockwork",
    "under an amphitheatre that whispers names",
    "on a coastline under three pale moons",
    "a lighthouse sweeping beams that paint colors, as kites hover without strings",
    "fog that parts into hallways, while footprints glow faintly behind them",
    "a horizon where two suns trade places, as staircases knit across the air",
    "buildings folding like paper cranes, while the moon appears in every window",
    "streetlights blooming into glass flowers, as trains arrive without tracks",
    "a bridge that arcs into space and back, as shadows detach and wander",
    "between sand dunes shaped like sleeping colossi",
    "against a stadium whose seats drift like lily pads",
    "ruins hovering one brick above the ground, while statues slowly turn their heads",
    "a shoreline of black glass pebbles humming, as doorways drift several inches off the ground",
    "alleyways that narrow into pinpoints of light, while statues slowly turn their heads",
    "across a causeway guarded by silent, winged lions",
    "a mangrove whose roots glow faintly, as flags ripple without wind",
    "between a city square paved with shifting clock faces",
    "inside a valley where stars fall upward",
    "a desert where shadows point in different directions, while birds trace geometric loops overhead",
    "underneath a chapel whose bells ring without sound",
    "pillars that drift like kites, as distant mountains fold like accordions",
    "between the rings of a distant planet shimmering overhead",
    "under mirrored corridors reflecting into infinity",
    "under a bridge that arcs into space and back",
    "a city sky filled with drifting auroras shaped like faces, while fountains pour upward",
    "along a ziggurat whose staircases slide",
    "underneath gusts that rearrange footprints",
    "a boulevard of doors that open to different skies, as distant mountains fold like accordions",
    "a mausoleum that breathes mist, while statues slowly turn their heads",
    "at the edge of streets bending upward like ribbons",
    "across a sundial casting multiple times at once",
    "under a city sky filled with drifting auroras shaped like faces",
    "beyond pillars that drift like kites",
    "inside a parking garage filled with shallow, silver water",
    "under streets bending upward like ribbons",
    "a theatre whose red curtains pour like water, while birds trace geometric loops overhead",
    "a clocktower shedding gears like petals, as doorways drift several inches off the ground",
    "at the edge of dawn light flickering like candle flames",
    "amid a river hanging in the air, unmoving",
    "between a forest of crystal trunks resonating softly",
    "beyond bridges curling like sleeping dragons",
    "along a halo sun circled by faint duplicates",
    "pillars that drift like kites, as staircases knit across the air",
    "streets bending upward like ribbons, as reflections move before the people do",
    "beneath ice crystals ringing like tiny bells",
    "beside a squall of luminous pollen",
    "a horizon where two suns trade places, as reflections move before the people do",
    "a square where rain falls upward as mist sinks, as kites hover without strings",
    "alleyways that narrow into pinpoints of light, as kites hover without strings",
    "a sky stitched with ribbons of amber vapor, while benches slide gently along the pavement",
    "a pier extending into a lake of rippling glass, as flags ripple without wind",
    "beyond a gate held up by singing chains",
    "within drizzle that paints temporary murals",
    "a theatre whose red curtains pour like water, as reflections move before the people do",
    "fields of wheat that turn to birds on the wind, while fountains pour upward",
    "across a reef of porcelain corals",
    "amid bridges curling like sleeping dragons",
    "above a chapel whose bells ring without sound",
    "beside a sky woven with unfamiliar constellations",
    "a halo sun circled by faint duplicates, as kites hover without strings",
    "underneath a reef of porcelain corals",
    "through a terminal where platforms slowly slide past each other",
    "through a marketplace where sound moves slower than light",
    "a neighborhood tiled in repeating copies of itself, while lanterns orbit like quiet planets",
    "a staircase made of moving shadows, while footprints glow faintly behind them",
    "across cliffs wrapped in slow, white fog ribbons",
    "a corridor of suspended stardust, as reflections move before the people do",
    "against a theatre whose red curtains pour like water",
    "a corridor of suspended stardust, as distant mountains fold like accordions",
    "ruins hovering one brick above the ground, as distant mountains fold like accordions",
    "across a ziggurat whose staircases slide",
    "a shrine surrounded by levitating offerings, while footprints glow faintly behind them",
    "clouds patterned like fingerprints, while statues slowly turn their heads",
    "clouds patterned like fingerprints, while benches slide gently along the pavement",
    "underneath the rings of a distant planet shimmering overhead",
    "within streets bending upward like ribbons",
    "within a bridge that arcs into space and back",
    "mist that pools like liquid silver, while statues slowly turn their heads",
    "a valley where stars fall upward, while streets rearrange their names",
    "a forest of crystal trunks resonating softly, while lanterns orbit like quiet planets",
    "beneath a museum with paintings that exhale fog",
    "along a stadium whose seats drift like lily pads",
    "ruins hovering one brick above the ground, as flags ripple without wind",
    "a sky tiled like a giant mosaic of galaxies, while lanterns orbit like quiet planets",
    "through a city sky filled with drifting auroras shaped like faces",
    "along a sky rumbling like a distant choir",
    "inside an amphitheatre that whispers names",
    "along row houses stitched together by laundry that never dries",
    "auroral curtains that trace silhouettes, as kites hover without strings",
    "along gusts that rearrange footprints",
    "beside a pier extending into a lake of rippling glass",
    "stone heads emerging from dew-wet grass, as flags ripple without wind",
    "a slow eclipse that never completes, while birds trace geometric loops overhead",
    "beneath a slow eclipse that never completes",
    "along a city square paved with shifting clock faces",
    "amid a sundial casting multiple times at once",
    "beneath a canyon of blue salt pillars",
    "beyond ice crystals ringing like tiny bells",
    "a plain where stones drift like soap bubbles, as staircases knit across the air",
    "between row houses stitched together by laundry that never dries",
    "a museum with paintings that exhale fog, as kites hover without strings",
    "within a canyon of blue salt pillars",
    "beyond a moonlit sea reflecting a second, inverted skyline",
    "inside a sundial casting multiple times at once",
    "beside a halo sun circled by faint duplicates",
    "across a valley where stars fall upward",
    "through subway tunnels filled with quiet, drifting balloons",
]


# Shot weights: favor closer shots a bit more to help faces (budget kept via low GEN_QUALITY)
SHOT_TYPES = [("wide shot", 0.25), ("medium shot", 0.45), ("close-up", 0.30)]

LIGHTING = ["daylight","overcast","evening light","fluorescent indoor light","front-lit","side-lit"]

ROLE_RULES = {
    "health": ["nurse","doctor","pharmacist","paramedic","emt","hospital administrator","nurse practitioner","midwife","health official","lab technician"],
    "education": ["teacher","student","professor","school principal","school counselor","pta"],
    "transport": ["bus driver","train operator","taxi","rideshare","pilot","flight attendant","ferry operator","station manager","dock worker","commuter"],
    "industry": ["factory","mechanic","construction","engineer","architect","warehouse","grid operator","power line","utility","recycling worker"],
    "agriculture": ["farmer","fisher"],
    "public_safety": ["firefighter","fire chief","police","police spokesperson","security guard","park ranger","conservationist","disaster relief"],
    "gov": ["politician","mayor","spokesperson","judge","minister","city planner","urban planner","election worker","charity director","ceo","researcher","scientist","weather official"],
    "media": ["news anchor","reporter","conference attendee"],
    "community": ["local resident","parent","volunteer","homeowner","neighbor","tourist","community organizer","voter","witness","athlete","runner","cyclist","sailor","veterinarian","animal shelter worker","museum curator","art teacher","librarian"]
}
EVENT_TAGS = {
    "health": ["hospital","clinic","vaccination","ward","patients","nurse","medical","lab"],
    "education": ["classroom","school","university","board","campus","graduates"],
    "transport": ["station","train","bus","airport","gate","terminal","platform","rail"],
    "industry": ["factory","assembly","construction","bridge","warehouse","equipment","prototype"],
    "agriculture": ["farm","farmers market","fair"],
    "public_safety": ["fire station","police","emergency","storm","wildfire","relief","shelter"],
    "gov": ["hearing","court","courthouse","town hall","press briefing","city hall","election"],
    "community": ["community","garden","vigil","charity","memorial","volunteers","food bank","neighborhood","aid"]
}
BACKDROP_TAGS = {
    "health": ["hospital"],
    "education": ["classroom","library"],
    "transport": ["station","platform","airport","stadium","rink"],
    "industry": ["factory","assembly","warehouse"],
    "agriculture": ["farm"],
    "public_safety": ["fire station","police"],
    "gov": ["courthouse","town hall"],
    "community": ["community center","garden","memorial"]
}

# ---------------- Grammar helpers -----------------
VOWELS = set("aeiou")

def is_plural_subject(s: str) -> bool:
    s_clean = s.strip()
    PLURALS = {"Children", "Tourists", "Graduates", "Volunteers"}
    if s_clean in PLURALS:
        return True
    if s_clean.endswith("s") and s_clean.lower() not in {"news"}:
        return True
    return False

def conjugate_third_person(base: str, plural: bool) -> str:
    if plural:
        return base
    if base.endswith(("s", "x", "z", "ch", "sh", "o")):
        return base + "es"
    if base.endswith("y") and len(base) > 1 and base[-2].lower() not in VOWELS:
        return base[:-1] + "ies"
    return base + "s"

def headline_tidy(s: str) -> str:
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Fix stray spaces before punctuation
    s = s.replace(" ,", ",").replace(" .", ".").replace(" :", ":")
    # Remove duplicated adjacent prepositions like "at the at" or "in in"
    PREP = r"(?:at|in|on|inside|under|amid|amidst|among|beside|near|by|underneath|around|within|across|through|as|during|while)"
    s = re.sub(rf"\b{PREP}\s+(?:the\s+)?{PREP}\b", lambda m: m.group(0).split()[0], s, flags=re.I)
    # Capitalize first character if needed
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # Ensure single terminal period
    s = s.rstrip()
    if s and s[-1] not in ".!?":
        s += "."
    return s

def role_to_subject(role: str) -> str:
    """Turn 'a nurse' -> 'Nurse' for subject slot."""
    r = role.strip().lower()
    r = re.sub(r"^(a|an)\s+", "", r)
    return r.capitalize() if " " not in r else " ".join(w.capitalize() for w in r.split())

def clean_event(event: str) -> str:
    """Keep articles; normalize phrasing."""
    e = event.strip()
    e = e.replace("answers questions on the sidelines", "answers questions")
    return e

# Actions in BASE FORM; conjugate later
ACTIONS_BASE = [
    "announce","discuss","review","inspect","outline","confirm","propose","signal","back","question",
    "promise","test","launch","detail","authorize","deny","file","appeal","clarify","commit",
    "approve","reject","pause","resume","accelerate","delay","greenlight","fast-track","shelve","revise",
    "adopt","table","withdraw","secure","negotiate","mediate","audit","survey","monitor","validate",
    "verify","calibrate","benchmark","brief","coordinate","mobilize","deploy","allocate","earmark","prioritize",
    "restructure","refinance","subsidize","regulate","standardize","ratify","penalize","rebate","waive","sanction",
    "evacuate","shelter","rescue","relocate","reopen","cordon","quarantine","decontaminate","vaccinate","screen",
    "triage","rehabilitate","recall","reissue","relabel","condemn","praise","commemorate","host","chair",
    "preside","facilitate","moderate","curate","stage","organize","pilot","trial","prototype","commission",
    "decommission","retrofit","overhaul","refactor","retool","repurpose","reconfigure","upgrade","patch","hotfix",
    "restore","stabilize","harden","fortify","mitigate","publish","retract","amend","correct","update",
    "summarize","forecast","nowcast","backcast"
]

# --------------- Dateline loader (required) ---------------
def load_datelines(max_items=5000) -> List[str]:
    paths = []
    if CITY_DATA_PATH:
        paths.append(Path(CITY_DATA_PATH))
    for cand in ["worldcities.csv", "cities1000.txt"]:
        p = Path(cand)
        if p.exists(): paths.append(p)
    for p in paths:
        try:
            if p.suffix.lower() == ".csv":
                with open(p, "r", encoding="utf-8", newline="") as f:
                    r = csv.DictReader(f)
                    names = []
                    for row in r:
                        name = row.get("city_ascii") or row.get("city") or row.get("name")
                        if not name: continue
                        names.append(name.strip())
                        if len(names) >= max_items: break
                if names:
                    return list(dict.fromkeys(names))
            else:
                with open(p, "r", encoding="utf-8") as f:
                    names=[]
                    for line in f:
                        parts = line.split("\t")
                        if not parts: continue
                        nm = parts[1].strip()
                        if nm: names.append(nm)
                        if len(names) >= max_items: break
                if names:
                    return list(dict.fromkeys(names))
        except Exception:
            pass
    raise SystemExit(
        "No dateline dataset found. Place 'worldcities.csv' (SimpleMaps) or 'cities1000.txt' (GeoNames) "
        "in the same folder as this script, or set CITY_DATA_PATH to an existing file."
    )

DATELINES = load_datelines(max_items=10000)
random.shuffle(DATELINES)

# --------------- Caption vocab (newswire style) ---------------
AFFECT_PAIRS = [
    ("smiles", "looks worried"),
    ("appears calm", "appears tense"),
    ("laughs", "looks solemn"),
    ("looks relieved", "appears anxious"),
    ("celebrates", "appears dejected"),
]

WHEN_FRAG = [
    "after safety audit","amid funding dispute","ahead of vote","following storm damage","during site visit",
    "after test run","amid staffing shortage","during inspection","after community complaints","as outreach continues",
    "after pilot program","amid contract talks","following late-night session","as crews work overnight",
    "after systems failure","as upgrades begin","after weekend outage","amid leadership change","ahead of deadline",
    "after briefing with aides","amid supply delays","after protocol review","amid rising demand","after board meeting",
    "as hearings resume","after panel report","as trial period starts","after tender announcement","amid regulatory review",
    "after third-party audit","amid budget cuts","after reconciliation talks","as temperatures climb","as storms approach",
    "after heatwave impacts","during cleanup effort","after evacuation drill","amid strike threat","after equipment checks",
    "during maintenance window","after minor incident","amid precautionary recall","after data breach notice",
    "amid cybersecurity review","after vote count","as ballots are verified","after by-election loss",
    "amid policy backlash","after stakeholder forum","as petition circulates","after training exercise",
    "amid hiring freeze","after recall announcement","amid consumer complaints","after route testing",
    "as timetable shifts","amid timetable confusion","after platform closure","amid track repairs",
    "after signal fault","as ferry service resumes","after runway inspection","during scheduled shutdown",
    "after night shift","amid staffing realignment","as funding window opens","after grant approval",
    "as interim report lands","after draft plan release","amid consultation period","after public submissions close",
    "amid land acquisition talks","after rezoning proposal","as tender opens","after bids received",
    "amid cost overrun debate","after feasibility study","as pilot expands","after phase-one completion",
    "amid design revisions","after prototype demo","as test results arrive","after procurement review",
    "amid court injunction","after settlement talks","as mediation continues","after compliance warning",
    "amid watchdog probe","after regulator’s letter","as fines loom","after code-of-conduct changes",
    "amid workplace review","after risk assessment","as safety checks expand","after evacuation order lifted",
    "amid flood warnings","after levee inspection","as fire bans extended","after containment update",
    "amid drought conditions","after water allocation change","as restrictions ease",
    "after harvest forecast","amid market volatility","after export delay","as tariffs discussed",
    "after exchange rate swing","amid cost-of-living pressure","after price cap debate",
    "as electricity demand peaks","after grid alert","amid rolling outages",
    "after substation failure","as turbines shut for service","after solar array fault",
    "amid community backlash","after neighborhood meeting","as volunteers mobilize",
    "after charity drive","amid donor shortfall","after grant rejection",
    "as university term starts","after semester break","amid exam week",
    "after scholarship announcement","as league season opens","after injury update",
    "as team returns home","after championship parade","amid coaching search",
    "after referee ruling","as training camp opens","after pre-season game",
    "amid travel disruption","after border change","as flights diverted",
    "after runway closure","amid baggage delays","as terminals crowd",
    "after marathon weekend","amid festival crowds","as holiday rush begins",
    "after weekend storms","amid heavy rainfall","as rivers peak",
    "after landslide warning","amid road closures","as detours lengthen",
    "after bridge inspection","amid tower crane removal","as site fencing expands",
    "after archaeological find","amid heritage listing","as public submissions surge"
]

OBJECTS_BY_TAG = {
    "health": [
        "new clinic hours","triage changes","equipment upgrades","protocol review","ventilator maintenance",
        "ICU capacity plan","ambulance response metrics","ER triage targets","elective surgery backlog",
        "vaccination timetable","cold-chain audit","pharmacy inventory rules","telehealth expansion",
        "mental health funding","nurse staffing ratios","PPE reserves","infection-control briefing",
        "outbreak response plan","ward refurbishment","pathology turnaround","laboratory accreditation",
        "paramedic rostering","medevac procedures","rural outreach clinics","air ambulance contract",
        "patient transfer protocols","radiology downtime","oncology trial enrollment","dialysis capacity",
        "maternity ward upgrades","defibrillator replacements","first-responder training","public health advisory",
        "contact tracing review","mask guidance update","clinic relocation","hospital visitor policy"
    ],
    "education": [
        "curriculum changes","facility upgrades","student support plans","STEM lab refurbishments",
        "digital literacy program","exam timetable","scholarship scheme","tutor hiring",
        "class size targets","school safety audit","cafeteria renovation","transport subsidies",
        "campus housing policy","research grant round","honorary degree list","graduation venue plan",
        "internship placements","industry partnerships","library opening hours","open day schedule",
        "assessment moderation","capstone showcase","plagiarism policy changes","learning analytics dashboard",
        "sports field resurfacing","student counseling expansion","language program intake","exchange quota",
        "lab safety training","field trip permissions","exam integrity software","lecturer recruitment"
    ],
    "transport": [
        "timetable changes","platform repairs","safety upgrades","signal faults","bridge load limits",
        "road resurfacing contract","airport slot allocation","runway lighting upgrade",
        "ferry timetable overhaul","bus lane expansion","bike-share pilot","level crossing removal",
        "ticketing app update","contactless rollout","fare freeze plan","night service trial",
        "depot modernization","fleet electrification","charging depot siting",
        "train carriage refurb","tram track renewal","flood detour routes","incident response drill",
        "road toll review","traffic signal retiming","port dredging schedule","container gate hours",
        "truck curfew extension","park-and-ride expansion","ride-hail pickup zones","microtransit pilot",
        "e-scooter rules","airport rail link study","bridge seismic retrofit","bus priority signals"
    ],
    "industry": [
        "line shutdown","safety audit results","hiring plans","shift changes","overtime policy",
        "maintenance backlog","inventory shortfall","supplier diversification","tooling upgrades",
        "robot cell calibration","QA nonconformance rate","ISO certification audit","warehouse automation",
        "forklift replacement","pallet tracking system","cold storage expansion","energy efficiency retrofit",
        "heat stress guidelines","lockout-tagout training","hazmat storage rules","ventilation upgrades",
        "noise exposure limits","industrial relations update","apprentice intake","contractor onboarding",
        "procurement framework","lean manufacturing sprint","just-in-time buffers","downtime analytics",
        "predictive maintenance trial","CMMS migration","safety stand-down","incident near-miss review",
        "RFID gate readers","dock scheduling platform","freight consolidation plan"
    ],
    "public_safety": [
        "response times","training plans","equipment checks","bushfire readiness","flood staging areas",
        "storm siren tests","evacuation route signage","volunteer pager rollout",
        "water rescue gear","thermal camera procurement","defensible space mapping",
        "strike team rosters","mutual aid compacts","ICS refresher course","AED network map",
        "urban search-and-rescue drill","sandbag stockpile","firebreak maintenance","smoke alarm program",
        "hydrant pressure tests","sirens coverage gaps","swiftwater unit training","lifeguard staffing",
        "emergency notification test","hazmat tabletop exercise","shelter capacity update",
        "cooling center hours","wildfire fuel loads","helicopter night ops","air quality shelter plan",
        "public alert wording","radio interoperability"
    ],
    "gov": [
        "funding package","policy draft","public consultation","select committee timetable",
        "legislative calendar","ombudsman referral","integrity commission brief",
        "anti-corruption safeguards","transparency register update","open data release",
        "procurement probity plan","infrastructure pipeline","rate rise model",
        "rezoning framework","urban growth boundary review","heritage overlay changes",
        "social housing tender","cost-of-living relief","rebate scheme","bill exposure draft",
        "regulatory impact statement","service backlog metrics","migration quotas",
        "visa processing update","border settings","federal-state agreement",
        "disaster relief appropriation","audit office findings","privacy law update",
        "digital ID roadmap","cloud procurement panel","election timetable","campaign finance returns"
    ],
    "agriculture": [
        "water allocations","harvest forecasts","market access","biosecurity controls",
        "quarantine station upgrade","spray drift guidelines","soil moisture readings",
        "drought relief grants","pest monitoring traps","locust watch alert",
        "shearing shed safety","abattoir inspection roster","vaccine cold chain",
        "export certification rules","traceability tags","commodity price outlook",
        "fertilizer supply update","irrigation roster","salinity mitigation trial","carbon farming pilot",
        "windbreak restoration","hail net funding","grain storage fumigation",
        "livestock transport curfew","farm vehicle permits","risk of foot-and-mouth",
        "veterinary cover gaps","bee colony counts","pollination service plan","regional packing shed"
    ],
    "community": [
        "volunteer drive","grant program","neighborhood project","street lighting upgrade",
        "accessible playground build","public art commission","library late hours",
        "community kitchen roster","youth sports bursary","seniors transport shuttle",
        "multicultural festival permit","night market trial","dog park fencing",
        "community garden plots","tree canopy targets","urban heat island study",
        "graffiti clean-up drive","noise abatement plan","traffic calming trial",
        "neighborhood watch meeting","waste sorting outreach","bulky waste pickup",
        "kerbside glass bins","green waste vouchers","shared e-bike hubs",
        "river foreshore revegetation","stormwater litter traps","street sweeping roster",
        "footpath repair blitz","open-streets weekend","heritage walk signage","wayfinding pilot"
    ]
}

# Patterns with optional em dash
PATTERNS = [
    "{D}: {S} {V} {O} {B}" if not USE_EM_DASH else "{D} — {S} {V} {O} {B}",
    "{D}: {S} {E} {B}" if not USE_EM_DASH else "{D} — {S} {E} {B}",
    "{S} {V} {O} {B}",
    "{S} {E} {B}",
    "{S} {V} {O} {B} {W}",
    "{D}: {S} {E} {B} {W}" if not USE_EM_DASH else "{D} — {S} {E} {B} {W}",
]

def infer_role_tags(role: str) -> Set[str]:
    role_l = role.lower()
    tags = set()
    for k, kws in ROLE_RULES.items():
        if any(kw in role_l for kw in kws):
            tags.add(k)
    if not tags: tags.add("community")
    return tags

def infer_text_tags(txt: str, table: Dict[str, List[str]]) -> Set[str]:
    t = txt.lower(); out = set()
    for k, kws in table.items():
        if any(kw in t for kw in kws): out.add(k)
    return out or {"community"}

def weighted_choice(items):
    r = random.random(); c = 0.0
    for name, w in items:
        c += w
        if r <= c: return name
    return items[-1][0]

# ---------------- Backdrop helpers -----------------
# For prompts (image generation), we like explicit prepositions:
_PREP_STARTS_PROMPT = tuple([
    "in ", "on ", "at ", "under ", "inside ", "amid ", "amidst ", "among ",
    "beside ", "near ", "by ", "over ", "underneath ", "around ", "within ", "across ", "through ",
    "as ", "during ", "while ", "atop ", "in front of ", "on top of ","between ", "along ", "against ", "above ", "below ", "beyond "
])

def loc_phrase(place: str) -> str:
    """Heuristic preposition for PROMPTS only; if already has one, return as-is."""
    p = place.strip()
    pl = p.lower()
    if pl.startswith(_PREP_STARTS_PROMPT):
        return p
    # falls back to at/in/on heuristics
    if any(k in pl for k in ["room","library","tunnel","tank","courtroom","server","museum","gallery",
                             "greenhouse","station","cave","aisle","classroom","forest","lab","cathedral",
                             "corridor","precinct","garage","dome","zoo","church","temple"]):
        prep = "in"
    elif any(k in pl for k in ["moon","glacier","iceberg","volcano","slope","court","surface","roof","ridge",
                               "bridge","runway","dam","pier","beach","floor","island","street","stairs","rings"]):
        prep = "on"
    else:
        prep = "at"
    return f"{prep} {p}"

_CAPTION_CONNECTORS = (
    "in ","on ","at ","under ","inside ","amid ","amidst ","among ",
    "beside ","near ","by ","over ","underneath ","around ","within ","across ","through ",
    "as ","during ","while ","atop ","in front of ","on top of ","inside of ","outside ","between ", "along ", "against ", "above ", "below ", "beyond "
)

def caption_backdrop(raw: str) -> str:
    """
    Normalize backdrop for CAPTIONS (not prompts):
    - If it already starts with a connector (preposition or 'as/during/while'), return as-is.
    - If it starts with a participle/verb-like phrase (e.g., 'floating', 'giving', 'standing', 'surrounded'),
      prefix 'while '.
    - Else, fall back to loc_phrase().
    """
    p = raw.strip()
    pl = p.lower()
    if pl.startswith(_CAPTION_CONNECTORS):
        return p
    # participial / verb-like opener -> use "while"
    if re.match(r"^(?:[a-z]+ing|being|surrounded|chased|facing|floating|speaking|giving|standing|walking|riding|holding|delivering|meeting|addressing)\b", pl):
        return f"while {p}"
    # otherwise, choose a sane preposition
    return loc_phrase(p)

# ---------------- Sentence builders -----------------
def newsroom_headline(role, event, back_caption, tagset, for_ta=False):
    subj = role_to_subject(role)          # singular
    e = clean_event(event)                # keep articles
    b = back_caption                      # already normalized
    d = random.choice(DATELINES)
    w = random.choice(WHEN_FRAG) if random.random() < 0.7 else ""
    v_base = random.choice(ACTIONS_BASE)
    v = conjugate_third_person(v_base, plural=False)
    tag_choice = random.choice(sorted(tagset)) if tagset else "community"
    o = random.choice(OBJECTS_BY_TAG.get(tag_choice, OBJECTS_BY_TAG["community"]))
    pat = random.choice(PATTERNS)
    core = pat.format(S=subj, V=v, O=o, B=b, E=e, D=d, W=w).strip()
    text = headline_tidy(core)

    fake_pos = []
    if for_ta:
        pos, neg = random.choice(AFFECT_PAIRS)
        aff = random.choice([pos, neg])
        text = re.sub(r"[.!?]\s*$", "", text) + f", {aff}."
        toks = text.split()
        fake_pos = list(range(len(toks)-len(aff.split())-1, len(toks)-1))
    return text, fake_pos

# --------------- Irrelevant headlines ----------------
IRREL_SUBJECTS = ["Marathon Runner","Chef","Violinist","Farmer","Spacecraft Engineer","Children",
                  "Tennis Player","Cyclist","Jazz Quartet","Tourists","Zookeeper","Surfer",
                  "Mountain Climber","Chess Grandmaster","Barista","Artist","Dancer","Sailor",
                  "Librarian","Florist","Archaeologist","Fashion Designer","DJ","Skateboarder"]
IRREL_ACTIONS_BASE  = [
    "win city race","plate signature dish","perform at gala","display fresh produce",
    "unveil prototype","paint mural","serve during tournament","repair flat tire",
    "rehearse before show","photograph landmark","feed animals","ride large wave",
    "reach rocky summit","make decisive move","pour latte art","open new collection",
    "rehearse routine","navigate rough seas","host reading","arrange bouquets",
    "debut capsule line","drop new single","land difficult trick"
]
IRREL_PLACES   = ["downtown","in concert hall","at weekend market","at technology expo",
                  "at community workshop","under stadium lights","on mountain trail","in small club",
                  "on windy coastline","at wildlife enclosure","on sunny beach","in gallery space",
                  "in rehearsal studio","near historic harbor","at public library","in flower shop",
                  "at pop-up venue","by city fountain","on plaza stage"]

def irrelevant_headline():
    s = random.choice(IRREL_SUBJECTS)
    a_base = random.choice(IRREL_ACTIONS_BASE)
    parts = a_base.split(" ", 1)
    verb_base = parts[0]
    rest = (" " + parts[1]) if len(parts) > 1 else ""
    plural = is_plural_subject(s)
    verb = conjugate_third_person(verb_base, plural=plural)
    p = random.choice(IRREL_PLACES)
    tail = random.choice(["", " in qualifier round", " after months of prep", " before capacity crowd"])
    hl = f"{s} {verb}{rest} {p}{tail}"
    hl = headline_tidy(hl)
    toks = hl.split()
    return hl, list(range(len(toks)))

# ---------------- Face detection & cropping -----------------
def detect_faces_info(img: Image.Image):
    """Return boxes, probabilities, and landmarks (if available)."""
    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    if boxes is None:
        return [], [], []
    out_boxes, out_probs, out_points = [], [], []
    for i, b in enumerate(boxes):
        if b is None:
            continue
        x1,y1,x2,y2 = [int(round(v)) for v in b.tolist()]
        x1=max(0,x1); y1=max(0,y1)
        x2=min(img.width,x2); y2=min(img.height,y2)
        if x2>x1 and y2>y1:
            out_boxes.append([x1,y1,x2,y2])
            out_probs.append(float(probs[i]) if probs is not None else 1.0)
            out_points.append(points[i] if points is not None else None)
    return out_boxes, out_probs, out_points

def face_quality_gate(img: Image.Image, boxes, probs, points, min_prob=0.92, min_side_px=140):
    """Reject frames where best face is low-confidence, too small, or eyes wildly misaligned."""
    if not boxes:
        return False
    best = max(range(len(boxes)), key=lambda i: probs[i] if i < len(probs) else 0.0)
    p = probs[best] if best < len(probs) else 0.0
    x1,y1,x2,y2 = boxes[best]
    side = min(x2-x1, y2-y1)
    if p < min_prob:
        return False
    if side < min_side_px:
        return False
    pts = points[best]
    if pts is not None:
        lx,ly = pts[0][0], pts[0][1]
        rx,ry = pts[1][0], pts[1][1]
        if abs(ly - ry) > 0.25 * max(1, abs(lx - rx)):  # eyes wildly misaligned
            return False
    return True

def _central_crop_to_ratio(img: Image.Image, w: int, h: int):
    iw, ih = img.size
    target = w / float(h)
    cur = iw / float(ih)
    if cur > target:
        nw, nh = int(round(ih * target)), ih
    else:
        nw, nh = iw, int(round(iw / target))
    x1 = (iw - nw) // 2
    y1 = (ih - nh) // 2
    return img.crop((x1, y1, x1 + nw, y1 + nh))

def crop_union(img: Image.Image, boxes: List[List[int]], w: int, h: int, expand=2.4):
    iw, ih = img.size
    if not boxes:
        return _central_crop_to_ratio(img, w, h).resize((w, h), Image.BICUBIC), []
    x1=min(b[0] for b in boxes); y1=min(b[1] for b in boxes)
    x2=max(b[2] for b in boxes); y2=max(b[3] for b in boxes)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*expand, (y2-y1)*expand
    target = w / float(h)
    if bw/bh > target: bh = bw/target
    else:              bw = bh*target
    x1n = int(max(0, cx - bw/2)); y1n = int(max(0, cy - bh/2))
    x2n = int(min(iw, cx + bw/2)); y2n = int(min(ih, cy + bh/2))
    crop = img.crop((x1n,y1n,x2n,y2n)).resize((w,h), Image.BICUBIC)
    sx, sy = w/float(x2n-x1n), h/float(y2n-y1n)
    new_boxes=[]
    for bx1,by1,bx2,by2 in boxes:
        tbx1=int(round((bx1-x1n)*sx)); tby1=int(round((by1-y1n)*sy))
        tbx2=int(round((bx2-x1n)*sx)); tby2=int(round((by2-y1n)*sy))
        tbx1=max(0,min(w,tbx1)); tbx2=max(0,min(w,tbx2))
        tby1=max(0,min(h,tby1)); tby2=max(0,min(h,tby2))
        if tbx2>tbx1 and tby2>tby1: new_boxes.append([tbx1,tby1,tbx2,tby2])
    return crop, new_boxes

def face_center_crop(img: Image.Image, boxes: List[List[int]], w: int, h: int):
    if not boxes:
        return _central_crop_to_ratio(img, w, h).resize((w, h), Image.BICUBIC), []
    iw, ih = img.size
    cx = sum((b[0]+b[2])/2 for b in boxes)/len(boxes)
    cy = sum((b[1]+b[3])/2 for b in boxes)/len(boxes)
    target = w/float(h); cur=iw/float(ih)
    if cur>target: nw=int(round(ih*target)); nh=ih
    else:          nw=iw;                   nh=int(round(iw/target))
    x1=max(0, min(iw-nw, int(round(cx - nw/2))))
    y1=max(0, min(ih-nh, int(round(cy - nh/2))))
    x2, y2 = x1+nw, y1+nh
    crop = img.crop((x1,y1,x2,y2)).resize((w,h), Image.BICUBIC)
    sx, sy = w/float(nw), h/float(nh)
    new_boxes=[]
    for bx1,by1,bx2,by2 in boxes:
        tbx1=int(round((bx1-x1)*sx)); tby1=int(round((by1-y1)*sy))
        tbx2=int(round((bx2-x1)*sx)); tby2=int(round((by2-y1)*sy))
        tbx1=max(0,min(w,tbx1)); tbx2=max(0,min(w,tbx2))
        tby1=max(0,min(h,tby1)); tby2=max(0,min(h,tby2))
        if tbx2>tbx1 and tby2>tby1: new_boxes.append([tbx1,tby1,tbx2,tby2])
    return crop, new_boxes

def apply_shot_crop(img: Image.Image, boxes: List[List[int]], shot: str, w: int, h: int):
    if shot == "wide shot":   return crop_union(img, boxes, w, h, expand=2.4)  # slightly tighter than 3.0
    if shot == "medium shot": return crop_union(img, boxes, w, h, expand=2.0)  # slightly tighter than 2.2
    return face_center_crop(img, boxes, w, h)

def save_jpeg(img: Image.Image, path: Path):
    img.save(path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)

def make_record(_id, rel_path, text, fake_cls, mtcnn_boxes, fake_image_box=None, fake_text_pos=None):
    return {
        "id": _id,
        "image": rel_path.replace("\\","/"),
        "text": text,
        "fake_cls": fake_cls,
        "fake_image_box": fake_image_box or [],
        "fake_text_pos": fake_text_pos or [],
        "mtcnn_boxes": mtcnn_boxes
    }
def append_jsonl(record: dict):
    """Append one record to metadata.jsonl immediately (crash-safe)."""
    META_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(META_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def snapshot_json(records: list):
    """Write a full pretty snapshot to metadata.json (periodic + on-exit)."""
    with open(META_SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# ---------------- Scene pool (controls absurd %; extra randomness) -----------------
def build_scene_pool(n=1200):
    """
    Build a pool of scene specs that workers will consume without repeats.
    Fixes: the absurd clause is now conditional on use_absurd, so your
    normal scenes stay normal when MISPLACED_BG_RATE < 1.0.
    """
    # shuffle vocab sources to increase randomness each run
    local_roles = ROLES[:]; random.shuffle(local_roles)
    local_events = EVENTS[:]; random.shuffle(local_events)
    local_backdrops = BACKDROPS[:]; random.shuffle(local_backdrops)
    local_absurd = ABSURD_BACKDROPS[:]; random.shuffle(local_absurd)

    seen, pool = set(), []
    ridx = eidx = bidx = aidx = 0

    while len(pool) < n:
        role = local_roles[ridx % len(local_roles)]; ridx += 1
        shot = weighted_choice(SHOT_TYPES)
        rt = infer_role_tags(role)

        # try a few times to find a fresh, valid combo
        for _ in range(80):
            event = local_events[eidx % len(local_events)]; eidx += 1
            et = infer_text_tags(event, EVENT_TAGS)

            use_absurd = (random.random() < MISPLACED_BG_RATE)

            if use_absurd:
                mismatch = True
                back_raw = local_absurd[aidx % len(local_absurd)]; aidx += 1
                back_prompt  = loc_phrase(back_raw)        # for image prompt
                back_caption = caption_backdrop(back_raw)  # for caption
            else:
                mismatch = False
                back_raw = local_backdrops[bidx % len(local_backdrops)]; bidx += 1
                back_prompt  = back_raw                    # already prepositional
                back_caption = back_raw

                # validity: normal scenes must be semantically matched
                bt = infer_text_tags(back_raw, BACKDROP_TAGS)
                good = (rt & et) and ((rt & bt) or (et & bt))
                if not good:
                    continue

            # props: don't "fix" absurd scenes
            props = []
            if use_absurd:
                if random.random() < 0.35: props.append("subject behaves normally despite catastrophic setting")
                if random.random() < 0.25: props.append("no protective gear; ordinary clothing")
                if random.random() < 0.20: props.append("physics-defying details (subtle)")
            else:
                if random.random() < 0.12: props.append("reporters nearby")
                if random.random() < 0.25: props.append("holding documents or a folder")
                if random.random() < 0.22: props.append("wearing safety gear appropriate to the site")

            prop_text = "; ".join(props) if props else "natural ambient details"
            deny = ", ".join(TEXT_BLACKLIST)
            light = random.choice(LIGHTING)
            face_hint = "; faces clearly visible, unobstructed; no sunglasses or opaque masks; " \
                        "primary face in crisp focus; features anatomically consistent; eyes aligned; no distortions"

            # text/logo avoidance (kept explicit so the model sees it)
            no_text = (
                f"avoid any readable text, letters, numbers, logos, banners, captions, scoreboards, watermarks; "
                f"signage abstract/blurred; do not render words such as {deny}"
            )

            # the key fix: switch the clause based on absurd vs normal
            if use_absurd:
                style_clause = (
                    "backdrop must be blatantly absurd, surreal, or impossible; "
                    "do not adapt wardrobe or props to make it plausible; "
                    "subject behaves completely normal as if nothing unusual is happening"
                )
            else:
                style_clause = (
                    "backdrop must be realistic and appropriate to the scene; "
                    "no disasters, no surreal or physics-defying elements"
                )

            prompt = (
                f"{role} {event} {back_prompt}; {shot}; editorial photojournalism; realistic; "
                f"1 to 3 people visible; avoid crowds{face_hint}; {light}; {prop_text}; {no_text}; {style_clause}"
            )

            key = (prompt, back_caption, shot, role, event)
            if key not in seen:
                seen.add(key)
                pool.append((prompt, back_caption, shot, role, event, mismatch))
                break

    random.shuffle(pool)  # extra shuffle to randomize worker consumption order
    return pool

# ---------------- OpenAI image -----------------
async def openai_image_async(prompt: str, counters=None) -> Image.Image:
    if MAX_IMAGE_CALLS is not None and counters is not None:
        if counters.setdefault("image_calls", 0) >= MAX_IMAGE_CALLS:
            raise RuntimeError("Reached MAX_IMAGE_CALLS")
        counters["image_calls"] += 1

    async def _call():
        r = await aclient.images.generate(
            model=IMG_MODEL,
            prompt=prompt,
            size=GEN_SIZE,
            quality=GEN_QUALITY,
            n=1,
        )
        b64 = r.data[0].b64_json
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    return await _with_backoff(_call)


# ---------------- ASYNC WORKERS -----------------
def random_scene():
    # pop from the pool for extra randomness without repeats
    try:
        return SCENE_POOL.pop()
    except IndexError:
        return random.choice(BACKUP_SCENES)

async def worker(lock, counters, records, pbar):
    goals = counters["goals"]
    # random jitter per worker to de-sync API calls
    await asyncio.sleep(random.uniform(0.0, 0.25))
    while True:
        async with lock:
            total_goal = sum(goals.values())
            total_done = sum(counters["done"].values())
            total_infl = sum(counters["inflight"].values())
            if total_done + total_infl >= total_goal:
                return
            if (MAX_IMAGE_CALLS is not None) and counters.get("image_calls", 0) >= MAX_IMAGE_CALLS:
                return
            # Pick a bucket randomly among those with remaining quota
            deficits = [k for k in ("lit","inv","irr") if counters["done"][k] + counters["inflight"][k] < goals[k]]
            if not deficits: return
            bucket = random.choice(deficits)
            counters["inflight"][bucket] += 1
            _id = counters["next_id"]; counters["next_id"] += 1

        # tiny jitter per task
        await asyncio.sleep(random.uniform(0.0, 0.15))

        prompt, back_caption, shot, role, event, mismatch = random_scene()

        # generate + face quality gating with limited retries
        attempt = 0
        try:
            img = await openai_image_async(prompt, counters=counters)
        except Exception:
            async with lock:
                counters["inflight"][bucket] -= 1
            continue

        while True:
            boxes_orig, probs, points = await asyncio.to_thread(detect_faces_info, img)
            count_ok = (MIN_FACES <= len(boxes_orig) <= MAX_FACES)
            quality_ok = face_quality_gate(img, boxes_orig, probs, points, min_prob=0.80, min_side_px=110)
            if count_ok and quality_ok:
                break
            attempt += 1
            if attempt >= MAX_REGEN_ATTEMPTS:
                async with lock:
                    counters["inflight"][bucket] -= 1
                # abandon this sample
                img = None
                break
            try:
                img = await openai_image_async(prompt, counters=counters)
            except Exception:
                attempt = MAX_REGEN_ATTEMPTS
        if img is None:
            continue

        # captions (local)
        if bucket == "irr":
            caption, fake_pos = irrelevant_headline()
            fake_cls = "text_swap"
        else:
            tags = infer_role_tags(role) | infer_text_tags(event, EVENT_TAGS)
            if bucket == "lit":
                caption, fake_pos = newsroom_headline(role, event, back_caption, tags, for_ta=False)
                fake_cls = "origin"
            else:
                caption, fake_pos = newsroom_headline(role, event, back_caption, tags, for_ta=True)
                fake_cls = "text_attribute"

        img_final, boxes_final = await asyncio.to_thread(apply_shot_crop, img, boxes_orig, shot, POST_W, POST_H)

        if STRICT_NO_TEXT and has_readable_text(img_final):
            img_final, still_bad = await asyncio.to_thread(scrub_text_if_detected, img_final, boxes_final)
            if still_bad:
                async with lock:
                    counters["inflight"][bucket] -= 1
                continue
        
        # --- DE-DUPE CHECK (strong) ---
        ph = await asyncio.to_thread(imagehash.phash, img_final)
        async with lock:
            phashes = counters.setdefault("phashes", set())
            if any((ph - old_h) <= PHASH_DISTANCE_MAX for old_h in phashes):
                counters["inflight"][bucket] -= 1
                continue
            # Caption-level de-dup, too (avoid repeated headlines)
            seen_texts = counters.setdefault("seen_texts", set())
            if caption in seen_texts:
                counters["inflight"][bucket] -= 1
                continue
            # Record both so future tasks compare against them
            phashes.add(ph)
            seen_texts.add(caption)


        out_dir = {"lit": DIR_LITERAL, "inv": DIR_INV, "irr": DIR_IRR}[bucket]
        out_path = out_dir / f"{uuid.uuid4().hex[:10]}.jpg"
        try:
            await asyncio.to_thread(save_jpeg, img_final, out_path)
        except Exception:
            async with lock:
                counters["inflight"][bucket] -= 1
            continue

        rel = out_path.relative_to(OUT_ROOT)
        rec = make_record(_id, str(rel), caption, fake_cls, boxes_final, fake_text_pos=fake_pos)

        async with lock:
            records.append(rec)
            append_jsonl(rec)  # <-- incremental write (JSONL) so we never lose progress

            counters["inflight"][bucket] -= 1
            counters["done"][bucket] += 1

            # periodic full snapshot every SNAP_EVERY samples
            total_now = counters["done"]["lit"] + counters["done"]["inv"] + counters["done"]["irr"]
            if total_now % SNAP_EVERY == 0:
                snapshot_json(records)

            pbar.update(1)

def list_jpgs(d: Path):
    return [p for p in d.glob("*.jpg") if p.is_file()]

def compute_phash(path: Path):
    try:
        im = Image.open(path).convert("RGB")
        return imagehash.phash(im)
    except Exception:
        return None

def preload_existing():
    """
    Scan OUT_ROOT for existing images + metadata.
    Returns:
      counts: {'lit':int,'inv':int,'irr':int}
      phashes: set of imagehashes from ALL saved images
      seen_texts: set of caption strings already used (from metadata.jsonl/json, if present)
      records: list of existing records (for snapshot carry-over)
    """
    counts = {"lit":0, "inv":0, "irr":0}
    phashes = set()
    seen_texts = set()
    records = []

    # Count files per bucket
    counts["lit"] = len(list_jpgs(DIR_LITERAL))
    counts["inv"] = len(list_jpgs(DIR_INV))
    counts["irr"] = len(list_jpgs(DIR_IRR))

    # pHash all existing images (quick + robust)
    for folder in [DIR_LITERAL, DIR_INV, DIR_IRR]:
        for p in list_jpgs(folder):
            h = compute_phash(p)
            if h is not None:
                phashes.add(h)

    if META_JSONL.exists():
        with open(META_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                records.append(obj)
                if "text" in obj and obj["text"]:
                    seen_texts.add(obj["text"])
    elif META_SNAPSHOT.exists():
        with open(META_SNAPSHOT, "r", encoding="utf-8") as f:
            arr = json.load(f)
            if isinstance(arr, list):
                records.extend(arr)
                for obj in arr:
                    if isinstance(obj, dict) and obj.get("text"):
                        seen_texts.add(obj["text"])


    return counts, phashes, seen_texts, records

async def amain():
        # --- preload existing so we only generate the remainder and never duplicate ---
    have_counts, have_phashes, have_texts, existing_records = preload_existing()

    # Adjust goals to remainder (never negative)
    goals = {"lit": TARGET_LITERAL, "inv": TARGET_INV_EMO, "irr": TARGET_IRRELEV}
    remainder = {k: max(0, goals[k] - have_counts.get(k, 0)) for k in goals}
    total = remainder["lit"] + remainder["inv"] + remainder["irr"]

    if total == 0:
        # Nothing to do; still write a snapshot and exit cleanly
        snapshot_json(existing_records)
        print("Done: 0 samples needed (targets already satisfied).")
        print(f"Split -> literal:{have_counts['lit']} invert:{have_counts['inv']} irrelevant:{have_counts['irr']}")
        print("image_calls:", 0)
        return

    records = []
    def _on_exit():
        with suppress(Exception):
            snapshot_json(records)

    atexit.register(_on_exit)

    def _sig_handler(signum, frame):
        _on_exit()
        raise SystemExit(f"Interrupted (signal {signum}); snapshot written.")

    with suppress(Exception):
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    
    # use the remainder for UI + pool sizing
    total = remainder["lit"] + remainder["inv"] + remainder["irr"]
    pbar = tqdm(total=total, desc=f"Building {total} DGM4-synthetic (seed={SEED})")

    lock = asyncio.Lock()

    # Seed counters with what's already done and known hashes/texts
    counters = {
        "done": {"lit":0, "inv":0, "irr":0},        # progress for THIS run only
        "inflight": {"lit":0, "inv":0, "irr":0},
        "next_id": 1 + (len(existing_records) if existing_records else 0),
        "phashes": have_phashes,                    # set of existing + new
        "seen_texts": have_texts,                   # set of existing + new captions
        "goals": remainder                          # per-bucket remainder for this run
    }
    # keep existing records so snapshots contain everything so far
    records = existing_records[:]

    # build pool sized for remainder
    TOTAL_TARGET_REMAINDER = total
    # ensure a healthy pool size to avoid prompt repeats
    pool_mult = 2.4 if TOTAL_TARGET_REMAINDER >= 400 else 4.0
    pool_size = max(600, int(TOTAL_TARGET_REMAINDER * pool_mult))
    global SCENE_POOL, BACKUP_SCENES
    SCENE_POOL = build_scene_pool(n=pool_size)
    BACKUP_SCENES = SCENE_POOL[: max(200, int(TOTAL_TARGET_REMAINDER * 0.10))]

    tasks = [asyncio.create_task(worker(lock, counters, records, pbar)) for _ in range(CONCURRENCY)]
    try:
        await asyncio.gather(*tasks)
    finally:
        snapshot_json(records)
    pbar.close()

    with open(OUT_ROOT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    # Final totals = previous + this run
    final_lit = have_counts["lit"] + counters["done"]["lit"]
    final_inv = have_counts["inv"] + counters["done"]["inv"]
    final_irr = have_counts["irr"] + counters["done"]["irr"]
    print(f"Done: {counters['done']['lit'] + counters['done']['inv'] + counters['done']['irr']} samples saved this run to {OUT_ROOT}")
    print(f"Split -> literal:{final_lit} invert:{final_inv} irrelevant:{final_irr}")
    print("image_calls:", counters.get("image_calls", 0))


if __name__ == "__main__":
    asyncio.run(amain())
