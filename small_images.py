import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic
from langsmith import Client

IMAGES_DIR = Path(__file__).parent / "images"

DATASET_NAME = "multimodal-image-descriptions"

EXAMPLES = [
    {
        "input": "Describe this image in detail.",
        "image_file": "city_skyline.jpg",
        "reference_output": "A city skyline with tall buildings and urban architecture.",
    },
    {
        "input": "What is the main subject of this image?",
        "image_file": "mountain_landscape.jpg",
        "reference_output": "A mountain landscape with snow-capped peaks and dramatic scenery.",
    },
    {
        "input": "Describe the mood and atmosphere of this image.",
        "image_file": "ocean_waves.jpg",
        "reference_output": "An ocean scene with waves, conveying a calm or powerful natural atmosphere.",
    },
    {
        "input": "What colors dominate this image?",
        "image_file": "forest_trail.jpg",
        "reference_output": "A forest trail dominated by green foliage and earthy brown tones.",
    },
    {
        "input": "Describe the lighting and time of day shown in this image.",
        "image_file": "desert_sunset.jpg",
        "reference_output": "A desert landscape during sunset with warm golden and orange lighting.",
    },
]


def create_dataset(ls_client: Client) -> str:
    """Create a LangSmith dataset with image attachments."""
    try:
        existing = ls_client.read_dataset(dataset_name=DATASET_NAME)
        ls_client.delete_dataset(dataset_id=existing.id)
        print(f"Deleted existing dataset: {DATASET_NAME}")
    except Exception:
        pass

    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Dataset for evaluating multimodal image description quality",
    )
    print(f"Created dataset: {DATASET_NAME} (id={dataset.id})")

    examples = []
    for ex in EXAMPLES:
        image_path = IMAGES_DIR / ex["image_file"]
        image_bytes = image_path.read_bytes()
        print(f"  Adding example: {ex['image_file']} ({len(image_bytes) / 1024 / 1024:.1f} MB)")

        examples.append({
            "id": uuid.uuid4(),
            "inputs": {
                "input": ex["input"],
                "image_file": ex["image_file"],
            },
            "outputs": {
                "answer": ex["reference_output"],
            },
            "attachments": {
                "image": {
                    "mime_type": "image/jpeg",
                    "data": image_bytes,
                },
            },
        })

    ls_client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Added {len(examples)} examples with image attachments")
    return dataset.id


def target_presigned_url(inputs: dict, attachments: dict) -> dict:
    """Approach 1: Use the presigned URL from LangSmith attachments.

    Pros: Simple, no extra API calls.
    Cons: Images must be under 5 MB (Claude's base64/URL limit).
    """
    client = anthropic.Anthropic()

    presigned_url = attachments["image"]["presigned_url"]

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": presigned_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": inputs["input"],
                    },
                ],
            }
        ],
    )

    answer = next(
        (block.text for block in response.content if block.type == "text"), ""
    )
    return {"answer": answer}


def target_files_api(inputs: dict, attachments: dict) -> dict:
    """Approach 2: Upload to Anthropic Files API, reference by file_id.

    Pros: Supports images up to 500 MB, avoids base64 overhead entirely.
    Cons: Extra upload step per image (could be cached/reused in production).
    """
    client = anthropic.Anthropic()

    image_reader = attachments["image"]["reader"]
    image_bytes = image_reader.read()
    mime_type = attachments["image"]["mime_type"]
    filename = inputs.get("image_file", "image.jpg")

    uploaded = client.beta.files.upload(
        file=(filename, image_bytes, mime_type),
    )

    try:
        response = client.beta.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "file", "file_id": uploaded.id},
                        },
                        {
                            "type": "text",
                            "text": inputs["input"],
                        },
                    ],
                }
            ],
            betas=["files-api-2025-04-14"],
        )
    finally:
        client.beta.files.delete(uploaded.id)

    answer = next(
        (block.text for block in response.content if block.type == "text"), ""
    )
    return {"answer": answer}


def llm_judge(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """LLM-as-judge evaluator using Claude to score image descriptions."""
    client = anthropic.Anthropic()

    prompt = f"""\
You are an expert evaluator assessing the quality of an image description.

The user asked: {inputs.get("input", "")}
The image file was: {inputs.get("image_file", "")}

Actual response:
{outputs.get("answer", "(no answer)")}

Expected response:
{reference_outputs.get("answer", "(no reference)")}

Evaluate the actual response on these criteria:
1. Relevance: Does the response actually answer the question asked?
2. Accuracy: Is the description consistent with what the expected output suggests?
3. Detail: Does the response provide a reasonable level of detail?
4. Coherence: Is the response well-structured and easy to understand?

First provide a brief assessment, then give a final score from 0.0 to 1.0.
You MUST end your response with exactly this format on the last line:
SCORE: <number between 0.0 and 1.0>"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = next(
        (block.text for block in response.content if block.type == "text"), ""
    )

    score = 0.5
    for line in reversed(response_text.strip().splitlines()):
        if line.strip().startswith("SCORE:"):
            try:
                score = float(line.strip().split("SCORE:")[1].strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                pass
            break

    return {
        "key": "image_description_quality",
        "score": score,
        "comment": response_text,
    }


def run_experiment(ls_client: Client, target_fn, experiment_prefix: str, description: str) -> None:
    """Run an offline experiment with an LLM-as-judge evaluator."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_prefix}")
    print(f"{'='*60}")

    results = ls_client.evaluate(
        target_fn,
        data=DATASET_NAME,
        evaluators=[llm_judge],
        experiment_prefix=experiment_prefix,
        description=description,
        max_concurrency=2,
    )

    print("\n--- Experiment Results ---")
    for result in results:
        example_input = result["example"].inputs
        run_output = result["run"].outputs or {}
        eval_results = result["evaluation_results"]["results"]
        print(f"\nImage: {example_input.get('image_file', 'N/A')}")
        print(f"Question: {example_input['input']}")
        answer = run_output.get("answer", "(error)")
        print(f"Answer: {answer[:150]}...")
        for er in eval_results:
            print(f"Score ({er.key}): {er.score}")
            if er.comment:
                print(f"Comment: {er.comment[:200]}")


def main():
    ls_client = Client()
    create_dataset(ls_client)

    # Approach 1: Presigned URL (simple, but limited to ~5 MB images)
    run_experiment(
        ls_client,
        target_fn=target_presigned_url,
        experiment_prefix="image-eval-presigned-url",
        description="Image descriptions via presigned URL (< 5 MB images)",
    )

    # Approach 2: Files API (supports images up to 500 MB)
    run_experiment(
        ls_client,
        target_fn=target_files_api,
        experiment_prefix="image-eval-files-api",
        description="Image descriptions via Anthropic Files API (up to 500 MB images)",
    )


if __name__ == "__main__":
    main()
