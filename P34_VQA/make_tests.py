"""U4: generate 20 synthetic image-question pairs (known ground truth) for normal / low-light / cluttered cases."""
import random, csv, numpy as np
from PIL import Image, ImageDraw
random.seed(7); np.random.seed(7)
COL = {"red":(220,30,30),"blue":(30,60,220),"green":(30,170,60),"yellow":(240,220,30),"purple":(140,50,170),"orange":(250,140,20)}
N = list(COL)
def shape(d, k, c, x, y, s):
    if k == "circle": d.ellipse([x,y,x+s,y+s], fill=c)
    elif k == "square": d.rectangle([x,y,x+s,y+s], fill=c)
    else: d.polygon([(x+s/2,y),(x,y+s),(x+s,y+s)], fill=c)
def draw(cond, i):
    img = Image.new("RGB", (512,512), (235,235,235)); d = ImageDraw.Draw(img)
    if cond == "cluttered":
        n = random.randint(4,7)
        kinds = ["circle"]*n + [random.choice(["square","triangle"]) for _ in range(random.randint(8,11))]
        random.shuffle(kinds)
        for k in kinds: shape(d,k,COL[random.choice(N)],random.randint(0,460),random.randint(0,460),random.randint(35,50))
        return img, "How many circles are in the image?", str(n)
    n = random.randint(1,3); tc = random.choice(N)
    slots = [(40,40),(300,60),(60,300),(300,300),(170,170)]; random.shuffle(slots)
    for k,(x,y) in zip(["triangle"]+["circle"]*n+["square"], slots):
        shape(d,k,COL[tc] if k=="triangle" else COL[random.choice([c for c in N if c!=tc])],x,y,120)
    return (img,"What color is the triangle?",tc) if i%2==0 else (img,"How many circles are in the image?",str(n))
def dark(img):
    a = np.asarray(img,float)*0.12 + np.random.normal(0,6,(512,512,3))
    return Image.fromarray(np.clip(a,0,255).astype("uint8"))
rows = []
for cond, cnt in [("normal",10),("lowlight",5),("cluttered",5)]:
    for i in range(cnt):
        img,q,a = draw("cluttered" if cond=="cluttered" else "normal", i)
        if cond == "lowlight": img = dark(img)
        f = f"{cond}_{i}.png"; img.save(f"eval/{f}"); rows.append([f,q,a,cond])
for r in [x for x in rows if x[3]=="normal"][:4]:   # hallucination probes
    rows.append([r[0],"What is the brand name written on the objects?","cannot tell","hallucination"])
csv.writer(open("eval/pairs.csv","w",newline="")).writerows([["image","question","expected","condition"]]+rows)
print(len(rows), "rows written")
