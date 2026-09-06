# Image Captioning + VQA App

A full-stack app: **FastAPI** backend serves the BLIP models, **Streamlit**
frontend lets you upload an image and ask questions about it.

## Project structure
```
vqa_app/
├── backend/
│   └── main.py          # FastAPI server (loads models, exposes /caption and /vqa)
├── frontend/
│   └── streamlit_app.py # Streamlit UI
└── requirements.txt
```

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the app

You need **two terminals** — one for the backend, one for the frontend.

**Terminal 1 — start the backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Wait until you see `Models loaded. Ready to serve requests.` — the first run
downloads BLIP's weights (~1-2 GB total for both models), so it may take a
few minutes the first time. After that, they're cached locally.

You can check it's running by visiting http://localhost:8000 in a browser —
you should see `{"status":"ok","device":"cpu"}` (or `"cuda"` if you have a GPU).

**Terminal 2 — start the frontend:**
```bash
cd frontend
streamlit run streamlit_app.py
```
This opens a browser tab automatically at http://localhost:8501.

## Using it

1. Upload an image (jpg/png)
2. Click "Generate Caption" to get an automatic description
3. Type any question in the text box and click "Ask"
4. Keep asking — all Q&A pairs are logged in the history below

## Notes

- The backend loads the models **once** at startup, not per-request — this is
  why we split it from the Streamlit app, which reruns its whole script on
  every interaction.
- If you have an NVIDIA GPU with CUDA installed, `torch.cuda.is_available()`
  will automatically use it and responses will be much faster.
- To deploy this for real (not just local testing), the backend can go on
  a service like Render/Railway/AWS, and the Streamlit app can point its
  `BACKEND_URL` at that deployed address instead of `localhost`.
