import os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_KEYS = [os.getenv(f"gemini_api_{i}") for i in range(1, 11)]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]




MODEL_HIERARCHY = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite"
]

# Use existing GEMINI_KEYS / MODEL_HIERARCHY from your app. If not defined, create empty lists.
try:
    _GEMINI_KEYS = GEMINI_KEYS
    _MODEL_HIERARCHY = MODEL_HIERARCHY
except NameError:
    _GEMINI_KEYS = []
    _MODEL_HIERARCHY = []

if not GEMINI_KEYS:
    raise RuntimeError("No Gemini API keys found. Please set them in your environment.")
