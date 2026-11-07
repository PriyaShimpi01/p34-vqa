# P34 · Visual Question Answering App
Ask questions about an uploaded image. Templates -> VQA (Groq vision LLM or BLIP-VQA) -> full-sentence answer.

## Run
Deployed app: `streamlit_app.py` on Streamlit Community Cloud (free). Local: `pip install -r requirements-local.txt` · set `GROQ_API_KEY` (and optional `HF_TOKEN`) as env vars / Space secrets · `streamlit run streamlit_app.py` (or `python app.py` for the Gradio version)
Evaluation: `python make_tests.py && python eval.py`

## Model card
- **Models**: Groq `meta-llama/llama-4-scout-17b-16e-instruct` (vision), `llama-3.3-70b-versatile` (answer expansion), `Salesforce/blip-vqa-base` (BSD-3), `stabilityai/sd-turbo` (non-commercial research licence).
- **Intended use**: educational VQA, accessibility aid for describing images. **Not for**: medical, legal, safety-critical decisions, or identifying people.
- **Limitations**: miscounts in cluttered scenes, weak in low light, may hallucinate unreadable text (see report numbers).
- **Safeguards**: regex guardrail refuses person-identification / sensitive-trait questions; model told to say "cannot tell".
- **Licence/IP**: model licences as above; the test images are synthetic and owned by the author. No API keys in repo.
