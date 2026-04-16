# Multimodal Image Evaluation with LangSmith

Demonstrates a LangSmith dataset and experiment flow that evaluates multimodal image descriptions using an LLM-as-judge. Compares two approaches for sending large images (4-5 MB) to Claude.

## Project Structure

```
.
├── main.py           # Script: dataset creation + both experiments end-to-end
├── notebook.ipynb    # Notebook: same logic, cell-by-cell for interactive use
├── images/           # 5 high-resolution JPEG images (3.5-5.2 MB each)
│   ├── city_skyline.jpg
│   ├── desert_sunset.jpg
│   ├── forest_trail.jpg
│   ├── mountain_landscape.jpg
│   └── ocean_waves.jpg
├── pyproject.toml
├── .env.example
└── README.md
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

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

### Script

```bash
uv run python main.py
```

Runs both experiments sequentially and prints results to stdout.

### Notebook

Open `notebook.ipynb` in Jupyter. Cells are designed to run top-to-bottom, or you can run individual experiments selectively.

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
