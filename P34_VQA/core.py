"""Core VQA logic: templates (U1), answer expansion (U2), BLIP/Groq VQA (U3), guardrail (Layer V)."""
import os, io, re, base64
from PIL import Image

VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")  # check console.groq.com/docs/models
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen/qwen3.8-27b")

# ---------- U1: question templates ----------
TEMPLATES = {
    "Object": ["What is the main object in the image?", "Is there a {object} in the image?"],
    "Counting": ["How many {object} are there?", "How many people are in the image?"],
    "Colour": ["What color is the {object}?", "What is the dominant color?"],
    "Spatial": ["What is to the left of the {object}?", "Where is the {object} located?"],
    "Text / OCR": ["What text is written in the image?", "What does the sign say?"],
    "Scene": ["Where was this photo taken?", "Is it day or night?"],
    "Accessibility": ["Describe this image for a blind user.", "Is it safe to cross here?"],
    "Unanswerable (hallucination test)": ["What is the brand name written on the objects?"],
}

# ---------- Guardrail ----------
BLOCK = re.compile(r"\b(who is (this|that|he|she)|identify (this|the|that) (person|man|woman)|name of (the|this) person|"
                   r"what (race|ethnicity|religion)|face recogni)", re.I)
def guard(q):
    if BLOCK.search(q):
        return "Refused: this app does not identify people or infer sensitive traits (race, religion) from images."
    return None

_client = None
def client():
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])  # set in Space secrets, never in code
    return _client

def b64(img):
    img = img.convert("RGB"); img.thumbnail((1024, 1024))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

# ---------- U3a: vision LLM via Groq (two prompt techniques compared) ----------
SYS = {
    "zero-shot": "You answer questions about an image. Reply with the shortest possible answer (1-3 words or a number). "
                 "If the answer cannot be determined from the image, reply exactly: cannot tell.",
    "chain-of-thought": "You answer questions about an image. First reason briefly step by step (count/scan carefully), "
                        "then end with a final line 'ANSWER: <1-3 words or a number>'. "
                        "If it cannot be determined from the image, the answer is: cannot tell.",
}
def vqa_groq(img, q, mode="zero-shot"):
    r = client().chat.completions.create(
        model=VISION_MODEL, temperature=0, max_tokens=300,
        messages=[{"role": "system", "content": SYS[mode]},
                  {"role": "user", "content": [{"type": "text", "text": q},
                                               {"type": "image_url", "image_url": {"url": b64(img)}}]}])
    t = r.choices[0].message.content.strip()
    m = re.search(r"ANSWER:\s*(.+)", t, re.I)
    return (m.group(1).strip() if m else t).rstrip(".")

# ---------- U3b: BLIP-VQA (BERT-based text encoder; local baseline) ----------
_blip = None
def vqa_blip(img, q):
    global _blip
    if _blip is None:
        from transformers import BlipProcessor, BlipForQuestionAnswering
        n = "Salesforce/blip-vqa-base"
        _blip = (BlipProcessor.from_pretrained(n), BlipForQuestionAnswering.from_pretrained(n))
    p, m = _blip
    out = m.generate(**p(img.convert("RGB"), q, return_tensors="pt"), max_new_tokens=10)
    return p.decode(out[0], skip_special_tokens=True)

# ---------- U2: expand terse answer into a full sentence (GPT-style LLM) ----------
def expand(q, terse):
    r = client().chat.completions.create(
        model=TEXT_MODEL, temperature=0.2, max_tokens=80,
        messages=[{"role": "system", "content": "Rewrite the question and short answer as ONE natural, complete sentence. "
                   "Do not add facts that are not in the answer. If the answer is 'cannot tell', say the image does not show enough to tell."},
                  {"role": "user", "content": f"Question: {q}\nShort answer: {terse}"}])
    return r.choices[0].message.content.strip()
