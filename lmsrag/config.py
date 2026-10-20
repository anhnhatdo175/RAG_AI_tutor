"""Central configuration for the LMS adaptive-profile RAG pipeline (v2, real data only)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RAW = ROOT / "data" / "raw"
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# ---- raw sources -----------------------------------------------------------
ASSISTMENTS_CSV = REPO / "adaptiverag" / "skill_builder_data.csv"
GSM8K_FILES = [RAW / "gsm8k_train.jsonl", RAW / "gsm8k_test.jsonl"]
MATH_FILES = [RAW / "math_test.jsonl"]
KHAN_DIR = ROOT / "data" / "khan"          # optional: drop Khan Academy exports here

# ---- artifacts ---------------------------------------------------------------
MAPPING_JSON = ART / "skill_mapping.json"
SKILL_CATALOG_JSON = ART / "skill_catalog.json"
CHUNKS_JSONL = ART / "corpus_chunks.jsonl"
EMB_NPY = ART / "chunk_embeddings.npy"
PROFILES_NPZ = ART / "profiles.npz"
RESULTS_JSON = ART / "results.json"

# ---- chunking (docx 2.3.1) ---------------------------------------------------
CHUNK_MAX_WORDS = 380       # ~512 tokens
CHUNK_OVERLAP_WORDS = 50

# ---- difficulty scale (Step 2) -----------------------------------------------
DIFFICULTY_LEVELS = (0.3, 0.6, 0.9)

# ---- profile (Step 3) --------------------------------------------------------
HALF_LIFE_OPPS = 8          # recency half-life, measured in opportunities on the skill
GAMMA = 0.7                 # weight of correctness vs. speed in w(t)
EMA_ALPHA = 0.25            # default mastery EMA rate (tuned on validation)
MIN_SKILL_DOCS = 20         # a skill is "retrievable" only if the corpus has >= this many chunks
MIN_DOCS_PER_LEVEL = 3      # ...and >= this many at each difficulty level it exposes

# ---- recommender (layer 4) ---------------------------------------------------
ZPD_DELTA = 0.10            # target stretch above current mastery
ZPD_SIGMA = 0.20
CF_NEIGHBOURS = 50
TARGET_SUCCESS = 0.70       # ZPD success target used by the evaluation

# ---- grade metadata (docx 2.4.2: grade(d) = target_grade) ---------------------------------------------------------
# ASSUMPTION - please confirm: none of the corpora carries a school grade, so `grade` is an ordinal band derived from the
# source and (for MATH) the difficulty level.  A record may override it with its own "grade" field (e.g. Khan jsonl).
GRADE_BANDS = ("primary_lower_secondary", "lower_secondary", "lower_upper_secondary", "upper_secondary")
GRADE_GSM8K = "primary_lower_secondary"                      # grade-school word problems
GRADE_BY_MATH_LEVEL = {1: "lower_secondary", 2: "lower_secondary", 3: "lower_upper_secondary",
                       4: "upper_secondary", 5: "upper_secondary"}
GRADE_DEFAULT = "lower_secondary"                            # Khan / lessons without an explicit grade
SUBJECT_DEFAULT = "Mathematics"                              # all bundled exercise corpora are math-focused
