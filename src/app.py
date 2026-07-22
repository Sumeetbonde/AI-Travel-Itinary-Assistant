import streamlit as st
import requests
import os

from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")


def parse_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        return response.text or "Unknown backend error"

    if isinstance(payload, dict):
        detail = payload.get("detail")
        message = payload.get("message")
        response_text = payload.get("response")
        return str(detail or message or response_text or payload)

    return str(payload)

st.title("🧭 AI Travel Itinerary Assistant")
st.markdown("Ask about any travel destination - we'll find the best suggestions for you!")
st.caption(f"Connected backend: {FASTAPI_URL}")

# Sidebar
with st.sidebar:
    st.subheader("📁 Upload Travel Guide")
    uploaded_file = st.file_uploader(
        "Upload a PDF travel guide (optional)",
        type="pdf"
    )

    if uploaded_file and st.button("Process Guide"):
        with st.spinner("Processing the travel guide..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                response = requests.post(
                    f"{FASTAPI_URL}/upload",
                    files=files,
                    timeout=180,
                )

                if response.ok:
                    payload = response.json()
                    result = payload.get("result", {})
                    st.success(f"'{uploaded_file.name}' processed successfully!")
                    if result:
                        st.info(
                            f"Pages: {result.get('pages')} | "
                            f"Chunks: {result.get('chunks')} | "
                            f"Collection: {result.get('collection')}"
                        )
                else:
                    st.error(f"Upload failed: {parse_error_message(response)}")

            except requests.RequestException as exc:
                st.error(f"Could not reach backend at {FASTAPI_URL}: {exc}")

# Main content
st.subheader("❓ Ask Your Question")

query = st.text_input(
    "Enter your travel question (e.g., Best places to visit in Paris):"
)

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/ask",
                    json={"query": query.strip()},
                    timeout=180,
                )

                if response.ok:
                    payload = response.json()
                    answer = payload.get("response", "No answer returned")
                    st.success("Here's your travel plan:")
                    st.markdown(answer)
                else:
                    st.error(f"Query failed: {parse_error_message(response)}")

            except requests.RequestException as exc:
                st.error(f"Could not reach backend at {FASTAPI_URL}: {exc}")

    else:
        st.warning("Please enter a query.")