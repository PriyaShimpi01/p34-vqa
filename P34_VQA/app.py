import os, time, numpy as np, gradio as gr
from PIL import Image
import core

def ask(img, q, model, mode, full):
    if img is None or not q.strip(): return "Upload an image and enter a question.", "", ""
    ref = core.guard(q)
    if ref: return ref, ref, "-"
    t = time.time()
    terse = core.vqa_blip(img, q) if model == "BLIP-VQA (local)" else core.vqa_groq(img, q, mode)
    ms = f"{(time.time()-t)*1000:.0f} ms"
    sentence = core.expand(q, terse) if full else ""
    return terse, sentence, ms

def fill(cat): return gr.update(choices=core.TEMPLATES[cat], value=core.TEMPLATES[cat][0])

def lowlight(img, factor):
    if img is None: return None
    a = np.asarray(img.convert("RGB"), float) * factor + np.random.normal(0, 6, (img.height, img.width, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))

def gen(prompt):
    from huggingface_hub import InferenceClient
    return InferenceClient(token=os.getenv("HF_TOKEN")).text_to_image(prompt, model="stabilityai/sd-turbo")

with gr.Blocks(title="P34 Visual Question Answering") as demo:
    gr.Markdown("# Visual Question Answering\nAsk anything about an image. Terse answer + full-sentence answer.")
    with gr.Tab("Ask"):
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Image")
                cat = gr.Dropdown(list(core.TEMPLATES), value="Object", label="Template category")
                tpl = gr.Dropdown(core.TEMPLATES["Object"], value=core.TEMPLATES["Object"][0], label="Template", allow_custom_value=True)
                q = gr.Textbox(label="Question (edit {object} placeholders)")
                model = gr.Radio(["Groq vision LLM", "BLIP-VQA (local)"], value="Groq vision LLM", label="Model")
                mode = gr.Radio(["zero-shot", "chain-of-thought"], value="zero-shot", label="Prompt technique (Groq only)")
                full = gr.Checkbox(True, label="Expand into full sentence (U2)")
                go = gr.Button("Answer", variant="primary")
            with gr.Column():
                o1 = gr.Textbox(label="Terse answer"); o2 = gr.Textbox(label="Full-sentence answer"); o3 = gr.Textbox(label="Latency")
        cat.change(fill, cat, tpl); tpl.change(lambda t: t, tpl, q)
        go.click(ask, [img, q, model, mode, full], [o1, o2, o3])
    with gr.Tab("Hard-case test images (U4)"):
        p = gr.Textbox(label="Prompt (sd-turbo)", value="a cluttered desk with many small objects, photo")
        b = gr.Button("Generate"); g = gr.Image(type="pil", label="Generated", interactive=False)
        b.click(gen, p, g)
        f = gr.Slider(0.05, 1, 0.12, label="Brightness factor"); b2 = gr.Button("Make low-light version")
        d = gr.Image(label="Low-light", interactive=False); b2.click(lowlight, [g, f], d)
    with gr.Tab("About"):
        gr.Markdown("Models: Groq vision LLM, BLIP-VQA-base, Llama-3.3-70B (expansion), sd-turbo. "
                    "Guardrail refuses person-identification. See README / model card.")
demo.launch()
