"""
Pundit Feynman LLM Client — 3-Stage Pipeline
Stage 1: Analyze   (images → structured JSON analysis)
Stage 2: Design    (analysis → implementation plan JSON)
Stage 3: Generate  (analysis + design → notebook cells JSON)
"""

import os
import json
import time
import re
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────
API_KEY = os.getenv("NVIDIA_API_KEY", "")
BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.5-397b-a17b")
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "8"))

# OCR Configuration
OCR_API_KEY = os.getenv("NVIDIA_OCR_API_KEY", "")
OCR_API_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"

# FLUX.1-schnell Image Generation
FLUX_API_KEY = os.getenv("NVIDIA_FLUX_API_KEY", "")
FLUX_API_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell"

MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=600.0,  # Explicit default timeout for the client
)


# ── Prompts ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert research engineer and educator who converts academic papers into "
    "clear, educational, executable Python code. You produce structured JSON output for "
    "each stage of the pipeline. When building toy implementations, you create REAL working code "
    "(PyTorch, Transformer layers, actual training loops) at reduced scale that "
    "runs on CPU. You prioritize faithful replication of the paper's architecture "
    "and algorithms while making the code deeply educational with clear explanations, "
    "using the Feynman technique to break down complex math into simple analogies, "
    "verbose logging, and insightful visualizations."
)

ANALYSIS_PROMPT = """Analyze this research paper text and return a JSON object with:
{
  "title": "exact paper title",
  "authors": ["author names"],
  "research_field": "e.g. NLP, Computer Vision, RL",
  "abstract_summary": "2-3 sentence plain English summary of the paper",
  "feynman_analogy": "A brilliant, everyday analogy that maps perfectly to the paper's core key_insight (e.g., comparing attention mechanisms to a cocktail party)",
  "feynman_core_concept": "Explain the paper's main idea as if teaching a bright 12-year-old, using the analogy above, in 3-5 sentences",
  "key_insight": "the core novel contribution in one sentence",
  "algorithms": [
    {
      "name": "algorithm name",
      "purpose": "what it does",
      "key_equations": ["important formulas in LaTeX notation"],
      "pseudocode_steps": ["step1", "step2"]
    }
  ],
  "architecture": {
    "type": "e.g. Transformer, CNN, GAN",
    "components": ["list of main components"],
    "data_flow": "description of how data flows through the model"
  },
  "datasets_mentioned": ["dataset names"],
  "implementation_requirements": {
    "frameworks": ["PyTorch"],
    "key_hyperparameters": {"param": "value"},
    "estimated_complexity": "low/medium/high for toy version"
  }
}

Return ONLY valid JSON, no markdown, no extra text."""

DESIGN_PROMPT = """Based on this paper analysis, create a toy implementation design that runs on CPU.
Return a JSON object with:
{
  "model_architecture": {
    "type": "architecture type",
    "embed_dim": 64,
    "num_layers": 2,
    "num_heads": 4,
    "vocab_size": 1000,
    "max_seq_len": 64,
    "components": [
      {
        "name": "component name",
        "class_name": "PythonClassName",
        "description": "what this component does",
        "key_params": {"param": "value"}
      }
    ]
  },
  "training_config": {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "num_epochs": 5,
    "batch_size": 16,
    "loss_function": "CrossEntropyLoss",
    "dataset_strategy": "synthetic generation approach"
  },
  "visualization_plan": [
    "loss curve",
    "attention heatmap",
    "sample predictions"
  ],
  "estimated_cells": 15,
  "code_structure": [
    {"section": "imports", "description": "required libraries"},
    {"section": "model", "description": "model architecture classes"},
    {"section": "data", "description": "synthetic data generation"},
    {"section": "training", "description": "training loop"},
    {"section": "evaluation", "description": "testing and visualization"}
  ]
}

Return ONLY valid JSON, no markdown, no extra text."""

GENERATE_PROMPT_TEMPLATE = """You are generating a Jupyter notebook from a paper analysis and implementation design.
Analysis: {analysis}
Design: {design}

Note: You are a 397B parameter model (Qwen 3.5) with 17B actively used parameters (MoE architecture).
This means you have deep expertise and vast knowledge. Use it to produce genuinely educational content.

Return a JSON array of notebook cells following this **exact 13-section structure**:

1. **Title & Overview** (markdown) — Paper title, authors, a one-paragraph summary of the paper.

2. **Table of Contents** (markdown) — Numbered list of all 13 sections. Each section name should be a clickable anchor link.

3. **The Feynman Explanation** (markdown) — A step-by-step explanation of the WHOLE paper using the Feynman technique. Break down the core algorithms, math, and architecture into the absolute simplest terms possible. Expand heavily on the `feynman_analogy` and `feynman_core_concept` from the analysis. Use relatable, everyday analogies for each major step so a beginner can intuitively grasp how the system works before seeing the code.

4. **Environment Setup** (code) — pip installs and imports. Include `torch`, `numpy`, `matplotlib`, and any other needed libraries.

5. **Configuration & Hyperparameters** (code) — A single config dict or dataclass with all hyperparameters. Add comments explaining each.

6. **Data Preparation** (code) — Synthetic dataset generation or loading. Must produce realistic dummy data matching the paper's domain.

7. **Model Architecture** (code) — Full PyTorch model implementation. Use `nn.Module` subclasses with detailed docstrings about each component. Include shape comments.

8. **Training Loop** (code) — Complete training loop with loss tracking, progress printing, and gradient clipping.

9. **Training Execution** (code) — Run the training and display results.

10. **Evaluation & Metrics** (code) — Run inference on test data and compute relevant metrics.

11. **Visualizations** (code) — Matplotlib charts: loss curves, attention heatmaps or feature maps, sample predictions.

12. **Key Takeaways** (markdown) — Bullet-point summary of what was learned, what would change at full scale, potential improvements.

13. **References** (markdown) — Paper citation, related work links, library documentation links.

Each cell in the JSON array must have:
{{"cell_type": "code" or "markdown", "source": "cell content as a string"}}

RULES:
- All code must be executable on CPU
- Use educational variable names and heavy commenting
- Include print() statements showing tensor shapes and intermediate results
- Follow the 13-section structure exactly
- Minimum 15 cells total
- The Feynman Explanation should be at least 300 words
- Return ONLY the JSON array, no markdown fences"""


# ── OCR extraction (NVIDIA NeMo Retriever OCR v1) ─────────────────────────

def extract_text_from_images(base64_images):
    """Extract text from paper page images using NVIDIA NeMo Retriever OCR API.
    Sends page images to the dedicated OCR model for fast, accurate extraction.
    Falls back to page-by-page if a batch request fails.
    """
    all_text = []
    headers = {
        "Authorization": f"Bearer {OCR_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    total = len(base64_images)
    print(f"  OCR: Processing {total} pages via NVIDIA NeMo Retriever...")

    for page_idx, img_b64 in enumerate(base64_images):
        print(f"    Page {page_idx + 1}/{total}...")

        payload = {
            "input": [
                {
                    "type": "image_url",
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            ],
            "merge_levels": ["paragraph"]
        }

        try:
            resp = requests.post(
                OCR_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()

            # Extract text from OCR response
            page_text = _parse_ocr_response(result, page_idx + 1)
            if page_text:
                all_text.append(page_text)

        except Exception as e:
            print(f"    \u26a0 OCR failed for page {page_idx + 1}: {e}")
            # Continue with remaining pages
            continue

    if not all_text:
        raise RuntimeError("OCR failed: No text extracted from any page")

    combined = "\n\n".join(all_text)
    print(f"  OCR complete: {len(combined)} chars from {len(all_text)}/{total} pages")
    return combined


def _parse_ocr_response(response_json, page_num):
    """Parse the NVIDIA OCR API response into clean text.
    Response format: {"data": [{"text_detections": [{"text_prediction": {"text": ..., "confidence": ...}}]}]}
    """
    texts = []
    try:
        for item in response_json.get("data", []):
            for detection in item.get("text_detections", []):
                pred = detection.get("text_prediction", {})
                text = pred.get("text", "").strip()
                confidence = pred.get("confidence", 0)
                # Only include text with reasonable confidence
                if text and confidence > 0.3:
                    texts.append(text)
    except Exception as e:
        print(f"    \u26a0 Error parsing OCR response for page {page_num}: {e}")
        return ""

    return "\n".join(texts)


# ── LLM Call with Retry ───────────────────────────────────────────────────

def call_with_retry(messages, max_tokens=4096, temperature=0.3, stream=False):
    """Call the LLM API with retry logic for transient errors."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            kwargs = dict(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=300,
            )
            if stream:
                kwargs["stream"] = True
                return client.chat.completions.create(**kwargs)
            else:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

        except Exception as e:
            error_str = str(e).lower()
            # Include "timeout" and "timed out" in retryable errors
            if any(kw in error_str for kw in ["429", "rate", "500", "503", "overloaded", "unavailable", "timeout", "timed out"]):
                last_error = e
                wait = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                print(f"  ⚠ Transient error: {e}. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_error}")


# ── JSON Parsing ──────────────────────────────────────────────────────────

def parse_llm_json(raw_text, step_name):
    """Parse JSON from LLM response, with cleanup and one repair attempt."""
    if raw_text is None:
        print(f"  ⚠ LLM returned None for {step_name}")
        return {}
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON parse failed in {step_name}. Attempting repair...")

    # Attempt auto-repair via LLM
    repair_prompt = (
        f"The following text was supposed to be valid JSON but has a syntax error:\n\n"
        f"{text[:6000]}\n\n"
        f"Error: {e}\n\n"
        f"Return ONLY the corrected valid JSON, nothing else."
    )
    repaired = call_with_retry(
        messages=[
            {"role": "system", "content": "You are a JSON repair tool. Return only valid JSON."},
            {"role": "user", "content": repair_prompt},
        ],
        max_tokens=max(len(text) // 2, 4096),
        temperature=0.1,
    )
    if repaired is None:
        raise ValueError(f"Could not repair JSON from {step_name} — LLM returned None")
    repaired = repaired.strip()
    if repaired.startswith("```"):
        repaired = repaired.split("\n", 1)[1]
    if repaired.endswith("```"):
        repaired = repaired[:-3]

    try:
        return json.loads(repaired.strip())
    except json.JSONDecodeError:
        # Last resort: try to extract JSON from the text
        json_match = re.search(r'[\[{].*[\]}]', repaired.strip(), re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Could not parse JSON from {step_name} even after repair.")


# ── Pipeline Stages ───────────────────────────────────────────────────────

def analyze_paper(raw_text):
    """Stage 1: Analyze extracted text into structured JSON."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{ANALYSIS_PROMPT}\n\n--- EXTRACTED PAPER TEXT ---\n\n{raw_text}"},
    ]
    raw = call_with_retry(messages, max_tokens=6144, temperature=0.2)
    return parse_llm_json(raw, "paper_analysis")


def design_implementation(analysis):
    """Stage 2: Create implementation design from analysis."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{DESIGN_PROMPT}\n\n--- PAPER ANALYSIS ---\n\n{json.dumps(analysis, indent=2)}"},
    ]
    raw = call_with_retry(messages, max_tokens=6144, temperature=0.2)
    return parse_llm_json(raw, "implementation_design")


def generate_notebook_cells_stream(analysis, design):
    """
    Stage 3: Generate notebook cells from analysis and design.
    Yields tokens from the LLM for live streaming in the UI.
    Finally yields the parsed cells list.
    """
    prompt = GENERATE_PROMPT_TEMPLATE.format(
        analysis=json.dumps(analysis, indent=2),
        design=json.dumps(design, indent=2),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Use streaming mode
    stream = call_with_retry(messages, max_tokens=65536, temperature=0.3, stream=True)
    full_response = []

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response.append(token)
            yield ("token", token)

    raw_text = "".join(full_response)
    result = parse_llm_json(raw_text, "notebook_cells")

    # Final logic to ensure we return a list of cells
    cells = []
    if isinstance(result, dict):
        cells = result.get("cells", [{"cell_type": "markdown", "source": json.dumps(result, indent=2)}])
    elif isinstance(result, list):
        cells = result
    else:
        cells = [{"cell_type": "markdown", "source": raw_text}]

    yield ("cells_final", cells)


# ── Streaming Pipeline ─────────────────────────────────────────────────────

def run_full_pipeline_stream(raw_text):
    """
    Orchestrates the full 3-stage pipeline.
    Yields SSE-formatted text events for the frontend code viewer.
    Returns final cells via the 'cells' key in the last event.

    Yields tuples of (event_type, data):
        ("text",  str)       — display text for the code viewer
        ("cells", list)      — final cells (only yielded once at end)
        ("analysis", dict)   — analysis metadata
        ("error", str)       — error message
    """
    try:
        # ── Stage 1: Analyze ──
        yield ("text", "\n  Analyzing Paper\n")
        yield ("text", "  " + "─" * 40 + "\n\n")

        analysis = analyze_paper(raw_text)

        if not analysis:
            yield ("text", "  Analysis returned empty. The LLM may have failed.\n\n")
            yield ("error", "Analysis returned empty result")
            return

        title = analysis.get("title", "Unknown Paper")
        field = analysis.get("research_field", "")
        insight = analysis.get("key_insight", "")
        algos = [a.get("name", "") for a in analysis.get("algorithms", [])]
        feynman_analogy = analysis.get("feynman_analogy", "")
        feynman_concept = analysis.get("feynman_core_concept", "")

        # Clean, minimal analysis output
        yield ("text", f"  {title}\n")
        yield ("text", f"  {field}\n\n")

        # The Feynman Explanation — the star of the show
        if feynman_analogy or feynman_concept:
            yield ("text", "  ─── The Feynman Explanation ───\n\n")
            if feynman_analogy:
                yield ("text", f"  {feynman_analogy}\n\n")
            if feynman_concept:
                yield ("text", f"  {feynman_concept}\n\n")

        if insight:
            yield ("text", f"  Key Insight: {insight}\n\n")

        yield ("text", "  Analysis complete.\n\n")

        yield ("analysis", {
            "title": title,
            "field": field,
            "insight": insight,
            "algorithms": algos,
            "feynman_analogy": feynman_analogy,
        })

        # ── Stage 2: Design ──
        yield ("text", "\n  Designing Implementation\n")
        yield ("text", "  " + "─" * 40 + "\n\n")

        design = design_implementation(analysis)
        if not design:
            design = {}

        arch = design.get("model_architecture", {})
        tc = design.get("training_config", {})
        yield ("text", f"  Architecture: {arch.get('type', 'N/A')}\n")
        yield ("text", f"  Training: {tc.get('optimizer', 'Adam')}, lr={tc.get('learning_rate', 0.001)}, {tc.get('num_epochs', 10)} epochs\n")
        yield ("text", "  Design complete.\n\n")

        # ── Stage 3: Generate (Now with LIVE STREAMING) ──
        yield ("text", "\n  Generating Notebook (Live Streaming)\n")
        yield ("text", "  " + "─" * 40 + "\n\n")

        cells = []
        for event_type, data in generate_notebook_cells_stream(analysis, design):
            if event_type == "token":
                # Yield raw tokens to the code viewer for "ghost-writing" effect
                yield ("text", data)
            elif event_type == "cells_final":
                cells = data

        code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
        md_cells = sum(1 for c in cells if c.get("cell_type") == "markdown")
        yield ("text", f"\n\n  ✅ Generation complete: {len(cells)} cells ({code_cells} code, {md_cells} markdown)\n")
        yield ("text", "  Notebook ready for download.\n")

        yield ("cells", cells)

    except Exception as e:
        yield ("error", str(e))


# ── Legacy compatibility ───────────────────────────────────────────────────
# Keep old function signatures working for backward compatibility

def extract_methodology(base64_images):
    """Legacy wrapper: extracts text from images."""
    return extract_text_from_images(base64_images)


# ── Visual Illustration (FLUX.1-schnell) ───────────────────────────────────

# System prompt for Qwen to craft image generation prompts
IMAGE_PROMPT_SYSTEM = """You are a world-class scientific illustrator and prompt engineer.
Your job: given a structured analysis of a research paper, write ONE prompt for an
AI image generator (FLUX) that will produce a clear, beautiful, academic-quality
visual illustration of the paper's CORE CONCEPT.

Rules:
1. Focus on the MAIN IDEA — the central algorithm, architecture, or mechanism.
2. Describe the visual layout precisely: shapes, arrows, labels, flow direction.
3. Use academic illustration style: clean lines, labeled components, white background.
4. Include spatial relationships: "on the left", "flowing into", "surrounded by".
5. Mention color coding for different components.
6. Do NOT include text/equations in the image — focus on visual metaphors.
7. Keep it to ONE paragraph, 80-120 words.
8. End with style keywords: "scientific diagram, educational poster, vector style,
   clean layout, professional, high resolution"

Return ONLY the prompt text, nothing else."""

def generate_concept_image(analysis):
    """
    Generate a visual illustration of a paper's core concept.
    Step 1: Qwen crafts a detailed, structured prompt from the analysis.
    Step 2: FLUX.1-schnell generates the image.
    Returns base64-encoded PNG string or None on failure.
    """
    if not FLUX_API_KEY:
        raise RuntimeError("NVIDIA_FLUX_API_KEY not set")

    # ── Step 1: Qwen → Image Prompt ──
    analysis_summary = json.dumps({
        "title": analysis.get("title", ""),
        "research_field": analysis.get("research_field") or analysis.get("field", ""),
        "key_insight": analysis.get("key_insight") or analysis.get("insight", ""),
        "algorithms": analysis.get("algorithms", []),
        "feynman_analogy": analysis.get("feynman_analogy", ""),
        "feynman_core_concept": analysis.get("feynman_core_concept", ""),
    }, indent=2)

    prompt_messages = [
        {"role": "system", "content": IMAGE_PROMPT_SYSTEM},
        {"role": "user", "content": f"Create an image generation prompt for this paper:\n\n{analysis_summary}"},
    ]

    print("  🎨 Generating image prompt via Qwen...")
    image_prompt = call_with_retry(prompt_messages, max_tokens=300, temperature=0.7)
    if not image_prompt:
        raise RuntimeError("Qwen returned empty image prompt")

    # Add preamble for FLUX to ensure academic quality
    full_prompt = (
        "A detailed, clean scientific illustration for an academic paper. "
        "Style: professional educational diagram, labeled components, "
        "modern flat vector design, white background, high contrast, "
        "color-coded sections, no text. "
        f"{image_prompt.strip()}"
    )
    print(f"  📝 FLUX prompt ({len(full_prompt)} chars): {full_prompt[:100]}...")

    # ── Step 2: FLUX.1-schnell → Image ──
    print("  🖼️  Calling FLUX.1-schnell...")
    headers = {
        "Authorization": f"Bearer {FLUX_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "prompt": full_prompt,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
    }

    response = requests.post(FLUX_API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"FLUX API error {response.status_code}: {response.text[:200]}")

    result = response.json()
    # FLUX returns {"image": "base64..."} or {"artifacts": [{"base64": "..."}]}
    image_b64 = None
    if "image" in result:
        image_b64 = result["image"]
    elif "artifacts" in result and len(result["artifacts"]) > 0:
        image_b64 = result["artifacts"][0].get("base64", "")

    if not image_b64:
        raise RuntimeError("FLUX returned no image data")

    print(f"  ✅ Image generated ({len(image_b64)} chars base64)")
    return image_b64
