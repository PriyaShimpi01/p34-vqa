import os, time, numpy as np, streamlit as st
from PIL import Image
try:
    for k in ("GROQ_API_KEY", "HF_TOKEN"):
        if k in st.secrets: os.environ[k] = st.secrets[k]
except Exception: pass
import core

st.set_page_config(page_title="P34 Visual Question Answering", layout="wide")
st.title("Visual Question Answering")
try:
    import transformers, torch; HAS_BLIP = True
except Exception: HAS_BLIP = False

tab1, tab2, tab3 = st.tabs(["Ask", "Hard-case test images (U4)", "About"])
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        f = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        if f: st.image(f)
        cat = st.selectbox("Template category", list(core.TEMPLATES))
        tpl = st.selectbox("Template", core.TEMPLATES[cat])
        q = st.text_input("Question (edit {object} placeholders)", value=tpl, key=f"q_{cat}_{tpl}")
        models = ["Groq vision LLM"] + (["BLIP-VQA (local)"] if HAS_BLIP else [])
        model = st.radio("Model", models, horizontal=True)
        mode = st.radio("Prompt technique (Groq)", ["zero-shot", "chain-of-thought"], horizontal=True)
        full = st.checkbox("Expand into full sentence (U2)", True)
        go = st.button("Answer", type="primary")
    with c2:
        if go:
            if not f or not q.strip(): st.warning("Upload an image and enter a question.")
            elif core.guard(q): st.error(core.guard(q))
            else:
                img = Image.open(f); t = time.time()
                with st.spinner("Thinking..."):
                    terse = core.vqa_blip(img, q) if model.startswith("BLIP") else core.vqa_groq(img, q, mode)
                    st.metric("Terse answer", terse)
                    if full: st.success(core.expand(q, terse))
                st.caption(f"Latency: {(time.time()-t)*1000:.0f} ms")
with tab2:
    st.write("Make a low-light version of any image to test hard cases.")
    g = st.file_uploader("Image to darken", type=["png", "jpg", "jpeg"], key="dk")
    fac = st.slider("Brightness factor", 0.05, 1.0, 0.12)
    if g:
        im = Image.open(g).convert("RGB")
        a = np.asarray(im, float) * fac + np.random.normal(0, 6, (im.height, im.width, 3))
        st.image(Image.fromarray(np.clip(a, 0, 255).astype("uint8")))
    st.divider()
    p = st.text_input("Optional: generate an image with sd-turbo (needs HF_TOKEN)", "a cluttered desk with many small objects, photo")
    if st.button("Generate"):
        try:
            from huggingface_hub import InferenceClient
            st.image(InferenceClient(token=os.getenv("HF_TOKEN")).text_to_image(p, model="stabilityai/sd-turbo"))
        except Exception as e: st.error(f"Generation unavailable: {e}")
with tab3:
    st.markdown("Models: Groq vision LLM, Llama-3.3-70B (expansion), BLIP-VQA-base (local only), sd-turbo. "
                "Guardrail refuses person-identification. See README / model card.")
