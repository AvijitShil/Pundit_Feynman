"""
Pundit Feynman — Research Paper to Executable Notebook
FastAPI backend with 3-stage AI pipeline, arXiv support, and SSE streaming.
"""

import os
import re
import uuid
import json
import time
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from utils.pdf_processor import process_pdf_to_base64
from utils.llm_client import extract_text_from_images, run_full_pipeline_stream, generate_concept_image
from utils.notebook_builder import build_notebook_from_cells

load_dotenv()

app = FastAPI(title="Pundit Feynman API", version="2.0")
os.makedirs("jobs", exist_ok=True)

# ── Concurrency limiter — max 3 simultaneous generations ──
_generation_semaphore = asyncio.Semaphore(3)


def _safe_remove(path, retries=3, delay=0.5):
    """Remove a file with retry for Windows file locking."""
    import time
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                print(f"  ⚠ Could not delete {path} (file locked)")


# ── Endpoint 1: Extract methodology from PDF upload ──────────────────────

@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    job_id = str(uuid.uuid4())
    pdf_path = f"jobs/{job_id}.pdf"

    # Save uploaded PDF
    with open(pdf_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        # Phase 1a: PDF → base64 images
        base64_images = process_pdf_to_base64(pdf_path)

        # Phase 1b: Vision extraction (batched)
        raw_text = extract_text_from_images(base64_images)

        # Save extracted text for Phase 2
        txt_path = f"jobs/{job_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # Clean up PDF
        _safe_remove(pdf_path)

        return {"job_id": job_id, "status": "extraction_complete", "pages": len(base64_images)}

    except Exception as e:
        print(f"  \u274c Extract error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(pdf_path):
            _safe_remove(pdf_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 1b: Extract from arXiv URL ──────────────────────────────────

@app.post("/api/extract-arxiv")
async def extract_arxiv(payload: dict):
    """Accept an arXiv URL, download the PDF, and run extraction."""
    import httpx

    arxiv_url = payload.get("url", "").strip()
    if not arxiv_url:
        raise HTTPException(status_code=400, detail="Missing 'url' field")

    # Extract paper ID from URL
    match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)', arxiv_url)
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Invalid arXiv URL. Expected format: https://arxiv.org/abs/XXXX.XXXXX"
        )

    paper_id = match.group(1)
    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    job_id = str(uuid.uuid4())
    pdf_path = f"jobs/{job_id}.pdf"

    try:
        # Download PDF from arXiv
        async with httpx.AsyncClient(follow_redirects=True) as http_client:
            print(f"  ⬇ Downloading PDF from arXiv: {pdf_url}")
            response = await http_client.get(pdf_url, timeout=30.0)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download PDF from arXiv: HTTP {response.status_code}"
                )

            # Save to disk
            with open(pdf_path, "wb") as f:
                f.write(response.content)

            size_mb = len(response.content) / (1024 * 1024)
            print(f"  ✅ Downloaded: {size_mb:.1f} MB")

        # Same pipeline as PDF upload
        base64_images = process_pdf_to_base64(pdf_path)
        raw_text = extract_text_from_images(base64_images)

        txt_path = f"jobs/{job_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        _safe_remove(pdf_path)

        return {
            "job_id": job_id,
            "status": "extraction_complete",
            "pages": len(base64_images),
            "arxiv_id": paper_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"  \u274c ArXiv extract error: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(pdf_path):
            _safe_remove(pdf_path)
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoint 2: Stream code generation (SSE) — 3-stage pipeline ──────────

@app.get("/api/generate_stream/{job_id}")
async def generate_stream(job_id: str):
    txt_path = f"jobs/{job_id}.txt"
    if not os.path.exists(txt_path):
        raise HTTPException(status_code=404, detail="Extraction not found. Run /api/extract first.")

    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"\n{'='*60}")
    print(f"  Starting 3-stage pipeline for job: {job_id}")
    print(f"  Text length: {len(raw_text)} chars")
    print(f"{'='*60}\n")

    def event_generator():
        notebook_path = f"jobs/{job_id}.ipynb"
        final_cells = None
        pipeline_success = False

        try:
            for event_type, data in run_full_pipeline_stream(raw_text):
                if event_type == "text":
                    payload = json.dumps({"text": data})
                    yield f"data: {payload}\n\n"

                elif event_type == "cells":
                    final_cells = data
                    print(f"  ✅ Pipeline produced {len(data)} cells")

                elif event_type == "analysis":
                    # Save analysis to disk for the /api/visualize endpoint
                    analysis_path = f"jobs/{job_id}_analysis.json"
                    try:
                        with open(analysis_path, "w", encoding="utf-8") as af:
                            json.dump(data, af)
                    except Exception:
                        pass
                    # Signal frontend that visualization is ready
                    yield f"data: {json.dumps({'analysis_done': True})}\n\n"

                elif event_type == "error":
                    err_msg = f"\n❌ Pipeline Error: {data}\n"
                    print(f"  ❌ Pipeline error: {data}")
                    payload = json.dumps({"text": err_msg})
                    yield f"data: {payload}\n\n"

        except Exception as e:
            err_msg = f"\n❌ Unexpected Error: {str(e)}\n"
            print(f"  ❌ Unexpected pipeline error: {e}")
            import traceback
            traceback.print_exc()
            err_payload = json.dumps({"text": err_msg})
            yield f"data: {err_payload}\n\n"

        # Build notebook from cells if we got them
        if final_cells and len(final_cells) > 0:
            try:
                build_notebook_from_cells(final_cells, notebook_path)
                pipeline_success = True
                print(f"  📓 Notebook saved: {notebook_path}")
            except Exception as e:
                print(f"  ❌ Failed to build notebook: {e}")
                err_payload = json.dumps({"text": f"\n❌ Failed to save notebook: {str(e)}\n"})
                yield f"data: {err_payload}\n\n"
        else:
            no_cells_msg = json.dumps({"text": "\n❌ Pipeline completed but no cells were produced. Check server logs for details.\n"})
            yield f"data: {no_cells_msg}\n\n"
            print(f"  ❌ No cells produced — notebook not saved")

        # Always send done event with status
        done_payload = json.dumps({"done": True, "success": pipeline_success})
        yield f"data: {done_payload}\n\n"

        # Only clean up extraction text on success
        if pipeline_success and os.path.exists(txt_path):
            os.remove(txt_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ── Endpoint 3: Download notebook ────────────────────────────────────────

async def cleanup_job_files(job_id: str):
    """Remove all job artifacts after download with a delay to ensure transfer."""
    await asyncio.sleep(10)  # Wait for download to start/finish
    for ext in [".pdf", ".txt", ".ipynb", "_analysis.json"]:
        path = f"jobs/{job_id}{ext}"
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


@app.get("/api/download/{job_id}")
async def download_notebook(job_id: str, background_tasks: BackgroundTasks):
    notebook_path = f"jobs/{job_id}.ipynb"
    if not os.path.exists(notebook_path):
        raise HTTPException(status_code=404, detail="Notebook not found")

    background_tasks.add_task(cleanup_job_files, job_id)

    return FileResponse(
        notebook_path,
        filename="pundit_feynman_notebook.ipynb",
        media_type="application/octet-stream",
    )


# ── Health check ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0", "pipeline": "3-stage"}


# ── Endpoint 5: Generate visual illustration ─────────────────────────────

@app.post("/api/visualize/{job_id}")
async def visualize_concept(job_id: str):
    """Generate a visual illustration of the paper's core concept."""
    print(f"\n[DEBUG] {time.strftime('%H:%M:%S')} 🎨 ROUTE HIT: /api/visualize/{job_id}")
    
    # Verify job id is sane
    if not job_id or job_id == "null" or job_id == "undefined":
        print(f"[DEBUG] ❌ ERROR: Received invalid Job ID: '{job_id}'")
        raise HTTPException(status_code=400, detail="Invalid Job ID received")

    analysis_path = f"jobs/{job_id}_analysis.json"
    if not os.path.exists(analysis_path):
        print(f"[DEBUG] ❌ ERROR: Analysis file does not exist: {analysis_path}")
        # List files in jobs to help debug
        print(f"[DEBUG] Files in jobs/: {os.listdir('jobs')}")
        raise HTTPException(status_code=404, detail=f"Analysis not found for job {job_id}")

    print(f"[DEBUG] 📂 Loading analysis JSON...")
    try:
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
    except Exception as e:
        print(f"[DEBUG] ❌ JSON ERROR: Could not parse {analysis_path}: {e}")
        raise HTTPException(status_code=500, detail="Corrupted analysis file")

    try:
        print(f"[DEBUG] 🖌️  Dispatching generation to threadpool for Job: {job_id}...")
        loop = asyncio.get_event_loop()
        image_b64 = await loop.run_in_executor(None, generate_concept_image, analysis)
        
        print(f"[DEBUG] ✅ SUCCESS: Generation finished for Job: {job_id}")
        return {"image": f"data:image/png;base64,{image_b64}"}
    except Exception as e:
        print(f"[DEBUG] ❌ GENERATION ERROR for Job {job_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ping")
async def ping():
    print("[DEBUG] 🏓 Ping received")
    return {"status": "ok", "message": "Pundit Feynman Backend is ALIVE"}


# ── Static files (MUST be last — catch-all) ──────────────────────────────

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
