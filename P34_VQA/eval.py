"""Run: python make_tests.py && python eval.py   -> correctness % per model/condition, latency, results.csv"""
import re, time, pandas as pd
from PIL import Image
import core
W = {"zero":"0","one":"1","two":"2","three":"3","four":"4","five":"5","six":"6","seven":"7","eight":"8","nine":"9"}
def norm(s): return [W.get(t,t) for t in re.findall(r"[a-z0-9]+", s.lower())]
def ok(pred, exp, cond):
    if cond == "hallucination": return bool(re.search(r"cannot|can't|unclear|no brand|no text|not visible|none|unknown", pred.lower()))
    return exp in norm(pred)
MODELS = {"BLIP-VQA": lambda i,q: core.vqa_blip(i,q),
          "Groq zero-shot": lambda i,q: core.vqa_groq(i,q,"zero-shot"),
          "Groq CoT": lambda i,q: core.vqa_groq(i,q,"chain-of-thought")}
df = pd.read_csv("eval/pairs.csv"); out = []
for _, r in df.iterrows():
    img = Image.open(f"eval/{r.image}")
    for name, fn in MODELS.items():
        t = time.time()
        try: p = fn(img, r.question)
        except Exception as e: p = f"ERROR {e}"; time.sleep(5)
        out.append(dict(model=name, condition=r.condition, image=r.image, question=r.question, expected=r.expected,
                        pred=p, correct=ok(p, str(r.expected), r.condition), sec=time.time()-t))
        time.sleep(1.5)  # free-tier rate limit
res = pd.DataFrame(out); res.to_csv("results.csv", index=False)
main = res[res.condition != "hallucination"]
print("\nOverall correctness % (20 pairs):\n", (main.groupby("model").correct.mean()*100).round(1))
print("\nBy condition %:\n", (main.pivot_table(index="model", columns="condition", values="correct", aggfunc="mean")*100).round(1))
print("\nHallucination probe (said 'cannot tell') %:\n", (res[res.condition=="hallucination"].groupby("model").correct.mean()*100).round(1))
print("\nAvg latency (s):\n", res.groupby("model").sec.mean().round(2))
print("\nFailures:\n", main[~main.correct][["model","image","question","expected","pred"]].to_string())
