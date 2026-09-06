"""
Streamlit Frontend for Image Captioning + VQA
-------------------------------------------------
Upload an image, see an auto-generated caption, then ask
as many custom questions about it as you like.

Talks to the FastAPI backend running on http://localhost:8000

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Image Q&A", page_icon="🖼️", layout="centered")

st.title("🖼️ Image Captioning & Visual Q&A")
st.write("Upload a photo, get a description, then ask it anything about the image.")

# --- Session state to remember the uploaded file + caption across reruns ---
if "caption" not in st.session_state:
    st.session_state.caption = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer) tuples

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Your uploaded image", use_container_width=True)

    # --- Generate caption automatically once per upload ---
    if st.button("Generate Caption"):
        with st.spinner("Looking at the image..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(f"{BACKEND_URL}/caption", files=files)
                response.raise_for_status()
                st.session_state.caption = response.json()["caption"]
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")

    if st.session_state.caption:
        st.success(f"**Caption:** {st.session_state.caption}")

    st.divider()
    st.subheader("Ask a question about the image")

    question = st.text_input("Your question", placeholder="e.g. Is there water in this picture?")

    if st.button("Ask") and question.strip():
        with st.spinner("Thinking..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"question": question}
            try:
                response = requests.post(f"{BACKEND_URL}/vqa", files=files, data=data)
                response.raise_for_status()
                answer = response.json()["answer"]
                st.session_state.history.append((question, answer))
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")

    # --- Show conversation history ---
    if st.session_state.history:
        st.divider()
        st.subheader("Q&A History")
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")
else:
    st.info("Upload an image above to get started.")
