
# Pundit Feynman 🧠

**Pundit Feynman** is a powerful research paper analyzer that converts complex PDFs into educational, executable PyTorch code—following the world-renowned **Feynman Technique**.

## 🚀 Deployment Instructions for Hugging Face Spaces

1.  **Create a New Space**: Choose **Docker** as the SDK.
2.  **Upload Files**: Upload the following:
    - `app.py`
    - `requirements.txt`
    - `Dockerfile`
    - `README.md`
    - `static/`
    - `utils/`
3.  **Add Secrets**: Go to **Settings > Variables and Secrets** and add your API keys:
    - `NVIDIA_API_KEY`: Your NVIDIA NIM API key.
    - `NVIDIA_FLUX_API_KEY`: (Optional) For visual illustrations.
4.  **Wait for Build**: Hugging Face will automatically build and deploy the Docker image.

## 🛠️ Tech Stack
- **Backend**: FastAPI
- **LLM**: NVIDIA NIM (Qwen-VL-Instruct / Qwen2-72B)
- **OCR**: NVIDIA NeMo Retriever
- **Image Gen**: FLUX.1-schnell
- **Frontend**: Vanilla HTML/CSS/JS (Beige & Serif Aesthetic)

