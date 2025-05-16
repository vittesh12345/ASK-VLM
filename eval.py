# evaluate_flamingo.py - 2025-05-09
# Evaluation script for comparing baseline and fine-tuned OpenFlamingo models

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader
from types import MethodType

# Set up logging
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[
        logging.FileHandler(f"logs/evaluation_{ts}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Configuration - using the exact paths you provided
IMAGES_DIR = "/root/.cache/kagglehub/datasets/nagasai524/mini-coco2014-dataset-for-image-captioning/versions/1/Images"
CAPTIONS_FILE = "/root/.cache/kagglehub/datasets/nagasai524/mini-coco2014-dataset-for-image-captioning/versions/1/captions.json"
BASELINE_MODEL_PATH = "./baseline_model"
FINETUNED_MODEL_PATH = "./flamingo_lora_output/best_model"


# Properly patch the Flamingo model for LoRA and generation
def properly_patch_flamingo(model):
    """
    Apply all necessary patches to the OpenFlamingo model for evaluation.
    """

    # Add generation method
    def _prepare_inputs_for_generation(self, input_ids=None, **kwargs):
        """Convert input_ids to lang_x for OpenFlamingo."""
        if input_ids is not None and "lang_x" not in kwargs:
            kwargs["lang_x"] = input_ids
        return kwargs

    # Add minimal config
    from transformers import PretrainedConfig

    if not hasattr(model, "config"):
        model.config = PretrainedConfig()
        model.config.model_type = "openflamingo"
        model.config.is_encoder_decoder = False

    # Add prepare_inputs_for_generation method
    model.prepare_inputs_for_generation = MethodType(
        _prepare_inputs_for_generation, model
    )

    # Add input_embedding accessors if needed
    if not hasattr(model, "get_input_embeddings"):

        def _get_input_embeddings(self):
            return self.lang_encoder.get_input_embeddings()

        def _set_input_embeddings(self, value):
            self.lang_encoder.set_input_embeddings(value)

        model.get_input_embeddings = MethodType(_get_input_embeddings, model)
        model.set_input_embeddings = MethodType(_set_input_embeddings, model)

    logger.info("Applied all necessary patches to Flamingo model")
    return model


# Evaluation dataset
class CaptioningEvalDataset(Dataset):
    def __init__(self, images_dir, captions_file, limit=None):
        self.images_dir = Path(images_dir)

        # Verify paths
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")

        if not os.path.exists(captions_file):
            raise ValueError(f"Captions file not found: {captions_file}")

        # Load captions
        logger.info(f"Loading captions from {captions_file}")
        with open(captions_file, "r") as f:
            try:
                captions_data = json.load(f)
                if isinstance(captions_data, dict) and "annotations" in captions_data:
                    # Handle COCO format
                    annotations = captions_data["annotations"]
                    images_info = {
                        img["id"]: img for img in captions_data.get("images", [])
                    }

                    # Extract info from COCO format
                    self.samples = []
                    for item in annotations[:limit]:
                        image_id = item.get("image_id")
                        if image_id is not None and image_id in images_info:
                            image_filename = images_info[image_id].get("file_name")
                            if image_filename:
                                image_path = os.path.join(
                                    self.images_dir, image_filename
                                )
                                if os.path.exists(image_path):
                                    self.samples.append(
                                        {
                                            "image_path": image_path,
                                            "caption": item.get("caption", ""),
                                        }
                                    )
                elif isinstance(captions_data, list):
                    # Handle simple list format
                    self.samples = []
                    for item in captions_data[:limit]:
                        if "image_id" in item and "caption" in item:
                            # Try to find the image filename
                            # Assume filename is the image_id with a jpg extension
                            image_filename = f"{item['image_id']}.jpg"
                            image_path = os.path.join(self.images_dir, image_filename)

                            # Check alternate formats if needed
                            if not os.path.exists(image_path):
                                image_filename = (
                                    f"COCO_train2014_{item['image_id']:012d}.jpg"
                                )
                                image_path = os.path.join(
                                    self.images_dir, image_filename
                                )

                            if os.path.exists(image_path):
                                self.samples.append(
                                    {
                                        "image_path": image_path,
                                        "caption": item["caption"],
                                    }
                                )
                else:
                    raise ValueError(f"Unrecognized captions format")
            except Exception as e:
                logger.error(f"Error parsing captions file: {e}")
                raise

        logger.info(f"Successfully loaded {len(self.samples)} image-caption pairs")

        # Print a few sample paths to verify
        if self.samples:
            logger.info("Sample image-caption pairs:")
            for i in range(min(3, len(self.samples))):
                logger.info(
                    f"  Image: {os.path.basename(self.samples[i]['image_path'])}"
                )
                logger.info(f"  Caption: {self.samples[i]['caption'][:50]}...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image if there's an error
            image = Image.new("RGB", (224, 224), color="white")

        return {"image": image, "image_path": image_path, "caption": sample["caption"]}


# Generation function for OpenFlamingo
def generate_captions_batch(
    model, processor, tokenizer, images, device, max_new_tokens=30
):
    """Generate captions for a batch of images"""
    # Process images with CLIP processor
    processed_images = [processor(img) for img in images]
    vision_x = (
        torch.stack(processed_images).unsqueeze(1).unsqueeze(1).to(device, dtype=dtype)
    )

    # Initialize with the prompt template
    prompt_template = "<image><|endofchunk|>"

    generated_captions = []

    # Process each image one by one
    for i in range(vision_x.size(0)):
        img_tensor = vision_x[i : i + 1]  # Take one image

        # Create the prompt
        inputs = tokenizer(prompt_template, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        # Keep track of generated tokens
        all_tokens = []

        # Generate tokens one by one
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Run the model
                outputs = model(
                    vision_x=img_tensor,
                    lang_x=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                )

                # Get next token logits from the last position
                next_token_logits = outputs[0][:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / 1.0  # temperature

                # Greedy decoding (take the most probable token)
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                # Add to generated
                all_tokens.append(next_token.item())

                # Add to the input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Stop on EOS token or max length
                if next_token.item() == tokenizer.eos_token_id:
                    break

        # Decode the generated caption
        generated_text = tokenizer.decode(all_tokens, skip_special_tokens=True).strip()
        generated_captions.append(generated_text)

    return generated_captions


# Metric functions
def compute_bleu(predictions, references):
    """Compute BLEU score"""
    if not predictions or not references:
        return {"bleu1": 0.0, "bleu4": 0.0}

    tokenized_preds = [pred.split() for pred in predictions]
    tokenized_refs = [
        [ref.split()] for ref in references
    ]  # BLEU expects a list of list of references

    smoothie = SmoothingFunction().method1
    bleu1 = corpus_bleu(
        tokenized_refs,
        tokenized_preds,
        weights=(1, 0, 0, 0),
        smoothing_function=smoothie,
    )
    bleu4 = corpus_bleu(
        tokenized_refs,
        tokenized_preds,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

    return {"bleu1": bleu1, "bleu4": bleu4}


def compute_rouge(predictions, references):
    """Compute ROUGE scores"""
    if not predictions or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        scores["rouge1"] += result["rouge1"].fmeasure
        scores["rouge2"] += result["rouge2"].fmeasure
        scores["rougeL"] += result["rougeL"].fmeasure

    # Average scores
    for key in scores:
        scores[key] /= max(len(predictions), 1)  # Avoid division by zero

    return scores


def compute_meteor(predictions, references):
    """Compute METEOR score"""
    if not predictions or not references:
        return {"meteor": 0.0}

    meteor_scores = []

    for pred, ref in zip(predictions, references):
        score = meteor_score([ref.split()], pred.split())
        meteor_scores.append(score)

    return {"meteor": sum(meteor_scores) / max(len(meteor_scores), 1)}


# Main evaluation function
def evaluate_model(model, processor, tokenizer, dataset, model_name, batch_size=4):
    """Evaluate a model on a dataset with multiple metrics"""
    logger.info(f"Evaluating {model_name} model...")
    model.eval()

    all_preds = []
    all_refs = []
    all_image_paths = []

    # Process in mini-batches
    total_samples = len(dataset)
    for batch_start in tqdm(
        range(0, total_samples, batch_size), desc=f"Generating captions - {model_name}"
    ):
        batch_end = min(batch_start + batch_size, total_samples)
        batch_indices = list(range(batch_start, batch_end))

        # Collect batch data
        batch_images = []
        batch_captions = []
        batch_paths = []

        for idx in batch_indices:
            sample = dataset[idx]
            batch_images.append(sample["image"])
            batch_captions.append(sample["caption"])
            batch_paths.append(sample["image_path"])

        try:
            # Generate captions for the batch
            batch_preds = generate_captions_batch(
                model, processor, tokenizer, batch_images, device
            )

            # Store results
            all_preds.extend(batch_preds)
            all_refs.extend(batch_captions)
            all_image_paths.extend(batch_paths)

            # Log occasionally
            if batch_start % (batch_size * 5) == 0 or batch_start == 0:
                for i in range(min(2, len(batch_preds))):
                    logger.info(
                        f"Sample {batch_start + i}: {os.path.basename(batch_paths[i])}"
                    )
                    logger.info(f"  Generated: {batch_preds[i]}")
                    logger.info(f"  Reference: {batch_captions[i]}")

        except Exception as e:
            logger.error(f"Error processing batch {batch_start}:{batch_end}: {e}")
            import traceback

            traceback.print_exc()

    # Calculate metrics
    results = {}

    # BLEU
    logger.info(f"Computing BLEU scores for {model_name}...")
    bleu_scores = compute_bleu(all_preds, all_refs)
    results.update(bleu_scores)

    # ROUGE
    logger.info(f"Computing ROUGE scores for {model_name}...")
    rouge_scores = compute_rouge(all_preds, all_refs)
    results.update(rouge_scores)

    # METEOR
    logger.info(f"Computing METEOR scores for {model_name}...")
    meteor_scores = compute_meteor(all_preds, all_refs)
    results.update(meteor_scores)

    return results, all_preds, all_refs, all_image_paths


def create_visualizations(metrics_df, samples_df, timestamp):
    """Create visualizations of the results"""
    # Convert percentage strings to floats for plotting
    metrics_df_plot = metrics_df.copy()

    for col in ["Baseline", "Fine-tuned", "Difference"]:
        metrics_df_plot[col] = metrics_df_plot[col].astype(float)

    metrics_df_plot["% Improvement"] = (
        metrics_df_plot["% Improvement"].str.rstrip("%").astype(float)
    )

    # Plot the metrics comparison
    plt.figure(figsize=(12, 8))

    metrics = metrics_df_plot["Metric"]
    baseline = metrics_df_plot["Baseline"]
    finetuned = metrics_df_plot["Fine-tuned"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, baseline, width, label="Baseline")
    rects2 = ax.bar(x + width / 2, finetuned, width, label="Fine-tuned")

    ax.set_title("Image Captioning Metrics: Baseline vs. Fine-tuned", fontsize=16)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.legend()

    # Add value labels on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(f"evaluation_results/metrics_comparison_{timestamp}.png")
    logger.info(
        f"Metrics visualization saved to evaluation_results/metrics_comparison_{timestamp}.png"
    )

    # Create a % improvement visualization
    plt.figure(figsize=(10, 6))
    improvement = metrics_df_plot["% Improvement"]
    colors = ["green" if x >= 0 else "red" for x in improvement]

    plt.bar(metrics, improvement, color=colors)
    plt.title("Percentage Improvement with Fine-tuning", fontsize=16)
    plt.ylabel("% Improvement", fontsize=12)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(improvement):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(f"evaluation_results/improvement_percentage_{timestamp}.png")
    logger.info(
        f"Improvement visualization saved to evaluation_results/improvement_percentage_{timestamp}.png"
    )


def main():
    # Print configuration
    logger.info("Evaluation Configuration:")
    logger.info(f"Baseline model path: {BASELINE_MODEL_PATH}")
    logger.info(f"Fine-tuned model path: {FINETUNED_MODEL_PATH}")
    logger.info(f"Images directory: {IMAGES_DIR}")
    logger.info(f"Captions file: {CAPTIONS_FILE}")

    # Verify paths
    for path in [IMAGES_DIR, CAPTIONS_FILE]:
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return

    # Load the OpenFlamingo model, processor, tokenizer
    try:
        from open_flamingo import create_model_and_transforms

        logger.info("Loading models and tokenizer...")

        # First create the tokenizer and processor which will be shared
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            cross_attn_every_n_layers=1,
        )

        # Configure tokenizer
        tokenizer.pad_token = "<PAD>"
        tokenizer.pad_token_id = 50279
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 0
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.bos_token_id = 0

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": ["<|endofchunk|>", "<image>"],
            "pad_token": "<PAD>",
            "eos_token": "<|endoftext|>",
            "bos_token": "<|endoftext|>",
        }
        tokenizer.add_special_tokens(special_tokens)

        # First load and patch the baseline model
        baseline_model, _, _ = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            cross_attn_every_n_layers=1,
        )

        # Patch it before loading checkpoint
        baseline_model = properly_patch_flamingo(baseline_model)

        # Load a checkpoint for the baseline model if available
        if os.path.exists(os.path.join(BASELINE_MODEL_PATH, "checkpoint.pt")):
            logger.info(
                f"Loading baseline model from {BASELINE_MODEL_PATH}/checkpoint.pt"
            )
            checkpoint = torch.load(
                os.path.join(BASELINE_MODEL_PATH, "checkpoint.pt"), map_location="cpu"
            )
            baseline_model.load_state_dict(checkpoint, strict=False)
        else:
            logger.info("Using default initialized model as baseline")

        baseline_model.to(device, dtype=dtype)
        logger.info("Baseline model loaded successfully")

        # Now load and patch a new model instance for the fine-tuned version
        finetuned_base_model, _, _ = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
            cross_attn_every_n_layers=1,
        )

        # CRITICAL: Apply patches BEFORE loading PEFT model
        finetuned_base_model = properly_patch_flamingo(finetuned_base_model)

        # Now wrap with PEFT
        from peft import PeftModel

        logger.info(f"Loading fine-tuned LoRA model from {FINETUNED_MODEL_PATH}")
        finetuned_model = PeftModel.from_pretrained(
            finetuned_base_model, FINETUNED_MODEL_PATH
        )
        finetuned_model.to(device, dtype=dtype)
        logger.info("Fine-tuned model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback

        traceback.print_exc()
        return

    # Load evaluation dataset
    try:
        eval_dataset = CaptioningEvalDataset(
            IMAGES_DIR,
            CAPTIONS_FILE,
            limit=30,  # Limit to 30 images for faster evaluation
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return

    # Evaluate baseline model
    try:
        baseline_results, baseline_preds, _, _ = evaluate_model(
            baseline_model, image_processor, tokenizer, eval_dataset, "Baseline"
        )
    except Exception as e:
        logger.error(f"Error evaluating baseline model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Evaluate fine-tuned model
    try:
        finetuned_results, finetuned_preds, references, image_paths = evaluate_model(
            finetuned_model, image_processor, tokenizer, eval_dataset, "Fine-tuned"
        )
    except Exception as e:
        logger.error(f"Error evaluating fine-tuned model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create results table
    results_table = {
        "Metric": [],
        "Baseline": [],
        "Fine-tuned": [],
        "Difference": [],
        "% Improvement": [],
    }

    for metric in baseline_results:
        baseline_value = baseline_results[metric]
        finetuned_value = finetuned_results[metric]
        diff = finetuned_value - baseline_value
        # Calculate percentage improvement
        pct_improvement = (diff / baseline_value * 100) if baseline_value != 0 else 0

        results_table["Metric"].append(metric)
        results_table["Baseline"].append(baseline_value)
        results_table["Fine-tuned"].append(finetuned_value)
        results_table["Difference"].append(diff)
        results_table["% Improvement"].append(pct_improvement)

    # Convert to DataFrame and display
    df = pd.DataFrame(results_table)
    # Format numeric columns
    for col in ["Baseline", "Fine-tuned", "Difference"]:
        df[col] = df[col].map(lambda x: f"{x:.4f}")
    df["% Improvement"] = df["% Improvement"].map(lambda x: f"{x:.2f}%")

    logger.info("\n=== EVALUATION RESULTS ===")
    logger.info(df.to_string(index=False))

    # Save sample predictions
    sample_results = []
    for i in range(min(10, len(references))):
        sample_results.append(
            {
                "Image": os.path.basename(image_paths[i]),
                "Reference": references[i],
                "Baseline": baseline_preds[i],
                "Fine-tuned": finetuned_preds[i],
            }
        )

    samples_df = pd.DataFrame(sample_results)
    logger.info("\n=== SAMPLE PREDICTIONS ===")
    logger.info(samples_df.to_string(index=False))

    # Save results to CSV
    os.makedirs("evaluation_results", exist_ok=True)
    df.to_csv(f"evaluation_results/metrics_{ts}.csv", index=False)
    samples_df.to_csv(f"evaluation_results/samples_{ts}.csv", index=False)

    # Create visualizations
    try:
        create_visualizations(df, samples_df, ts)
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

    logger.info(f"\nResults saved to evaluation_results/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
