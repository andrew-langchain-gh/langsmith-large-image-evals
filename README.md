# Multimodal Image Evaluation with LangSmith

Demonstrates a LangSmith dataset and experiment flow that evaluates multimodal image descriptions using an LLM-as-judge. Two scripts, two datasets, two size regimes:

- **`small_images.py`** — small/medium images (3.5–5.2 MB). Runs both the presigned-URL and Files API approaches side by side.
- **`large_images.py`** — large images (>10 MB, ~20 MB each). Runs only the Files API approach; presigned URLs can't handle images above Claude's ~5 MB URL limit.

## Project Structure

```
.
├── small_images.py        # Small-image script: both experiments end-to-end
├── large_images.py        # Large-image script: Files API experiment only
├── notebook.ipynb         # Notebook: same logic as large_images.py, cell-by-cell
├── images/                # Source images
│   ├── city_skyline.jpg       # 5.2 MB
│   ├── desert_sunset.jpg      # 5.2 MB
│   ├── forest_trail.jpg       # 4.2 MB
│   ├── mountain_landscape.jpg # 4.7 MB
│   ├── ocean_waves.jpg        # 3.5 MB
│   ├── huge_nebula.png        # 20.3 MB
│   ├── huge_canyon.png        # 20.7 MB
│   └── huge_reef.png          # 21.9 MB
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

[Git LFS](https://git-lfs.com/) is required to clone this repo — the ~20 MB PNGs under `images/` are stored via LFS. Without it, those files will appear as small text pointer files instead of the actual images. On macOS:

```bash
brew install git-lfs
git lfs install  # one-time per-user setup
```

Then clone (or `git lfs pull` if you already cloned without LFS), and install Python deps:

```bash
uv sync
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
|---|---|---|
| `LANGSMITH_API_KEY` | Yes | LangSmith dataset and experiment access |
| `ANTHROPIC_API_KEY` | Yes | Claude API for image descriptions and LLM judge |
| `OPENAI_API_KEY` | No | Only needed if using openevals with OpenAI models |
| `LANGSMITH_PROJECT` | No | LangSmith project name (defaults to `multimodal-image-slowness`) |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |

## Usage

### Small/medium images (`small_images.py`)

```bash
uv run python small_images.py
```

Creates the `multimodal-image-descriptions` dataset from the five JPEGs and runs both experiments sequentially:

1. **`image-eval-presigned-url`** — Claude reads the LangSmith-provided presigned URL directly. Works because each JPEG is under the ~5 MB URL limit.
2. **`image-eval-files-api`** — Each image is uploaded to Anthropic's Files API and referenced by `file_id`.

### Large images (`large_images.py`)

```bash
uv run python large_images.py
```

Creates a separate `multimodal-large-image-descriptions` dataset from the three ~20 MB PNGs and runs only the Files API experiment (`large-image-eval-files-api`). The presigned-URL approach is skipped because Claude rejects URL-referenced images above ~5 MB.

Two size caps matter here:

- **LangSmith Cloud: 25 MB per request.** `create_examples` is called one example at a time, since three 20 MB attachments in a single call would exceed the cap (and CloudFront returns `403` instead of `413`).
- **Claude Files API: 500 MB per file.** Plenty of headroom for images up to that size.

### Notebook

Open `notebook.ipynb` in Jupyter. Cells mirror `large_images.py` (large-image workflow only) and are designed to run top-to-bottom.

## How It Works

### Dataset

Creates a LangSmith dataset (`multimodal-image-descriptions`) with 5 examples. Each example has:

- **inputs**: `{"input": "<question about the image>", "image_file": "<filename>"}`
- **outputs**: `{"answer": "<reference description>"}`
- **attachments**: the JPEG image file (uploaded as binary via LangSmith's attachment API)

### Two Approaches for Sending Images to Claude

Claude's Messages API has a **5 MB limit** on base64-encoded and URL-referenced images. Since our images are 3.5-5.2 MB on disk, base64 encoding (which adds ~33% overhead) would push them over the limit. Two alternatives:

#### Approach 1: Presigned URL (`target_presigned_url`)

LangSmith provides a presigned URL for each attachment. Pass it directly to Claude's `image.source.url` field.

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "url", "url": presigned_url}},
            {"type": "text", "text": "Describe this image"},
        ],
    }],
)
```

- **Pros**: Simple, no extra API calls
- **Cons**: Still subject to Claude's ~5 MB URL image limit

#### Approach 2: Anthropic Files API (`target_files_api`)

Upload the image to Anthropic's Files API, then reference it by `file_id`. Supports files up to **500 MB**.

```python
uploaded = client.beta.files.upload(file=(filename, image_bytes, mime_type))

response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "file", "file_id": uploaded.id}},
            {"type": "text", "text": "Describe this image"},
        ],
    }],
    betas=["files-api-2025-04-14"],
)
```

- **Pros**: Handles images far beyond 5 MB, no base64 overhead
- **Cons**: Extra upload/delete API calls per image, requires beta flag

### LLM-as-Judge Evaluator

A custom evaluator (`llm_judge`) that uses Claude to score each image description on four criteria:

1. **Relevance** - Does the response answer the question?
2. **Accuracy** - Is it consistent with the expected output?
3. **Detail** - Does it provide reasonable detail?
4. **Coherence** - Is it well-structured?

Returns a score from 0.0 to 1.0 with a written assessment as the comment.

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API client |
| `langsmith` | Dataset creation, experiment runner |
| `openevals` | Pre-built evaluator utilities (available but not used in final version) |
| `python-dotenv` | Load `.env` file |
