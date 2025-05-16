#!/usr/bin/env python
# flamingo_peft_train_fixed.py  –  2025‑05‑09
#
# OpenFlamingo + PEFT‑LoRA fine‑tuning, now with a custom
# `resize_embeddings()` that works for the plain Flamingo module.

import os, sys, json, gc, random, logging, traceback
from datetime import datetime
from types import MethodType

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from transformers import PretrainedConfig, get_linear_schedule_with_warmup
from PIL import Image
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
import numpy as np
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from collections import defaultdict
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    handlers=[
        logging.FileHandler(f"logs/train_{ts}.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Seeds set to {seed}")


def resize_embeddings(model, tokenizer):
    """
    Manually grow the language‑side embedding & lm_head to match the
    tokenizer's new vocab size.
    """
    old_emb = model.lang_encoder.get_input_embeddings()
    old_tokens, dim = old_emb.weight.shape
    new_tokens = len(tokenizer)
    if new_tokens <= old_tokens:
        return  # nothing to do

    logger.info(f"Resizing embeddings: {old_tokens} → {new_tokens}")
    new_emb = torch.nn.Embedding(new_tokens, dim)
    new_emb.weight.data[:old_tokens] = old_emb.weight.data
    torch.nn.init.normal_(new_emb.weight.data[old_tokens:], std=0.02)
    model.lang_encoder.set_input_embeddings(new_emb)

    # If the backbone exposes an lm_head tied to embeddings, grow it too
    if hasattr(model.lang_encoder, "lm_head"):
        old_lm = model.lang_encoder.lm_head
        if old_lm.weight.shape[0] == old_tokens:
            new_lm = torch.nn.Linear(dim, new_tokens, bias=False)
            new_lm.weight.data[:old_tokens] = old_lm.weight.data
            torch.nn.init.normal_(new_lm.weight.data[old_tokens:], std=0.02)
            model.lang_encoder.lm_head = new_lm


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


import os, torch, json, gc
from datetime import datetime
from types import MethodType
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------------------------------------------------
# ── helper: make Flamingo accept `input_ids` ───────────────────────
# ------------------------------------------------------------------
# -----------------------------------------------------
# helper ─ patch Flamingo so PEFT is happy
# -----------------------------------------------------
# ---------------------------------------------------------------
# helper ─ make an OpenFlamingo model look like a HF text model
# ---------------------------------------------------------------
from types import MethodType
from transformers import PretrainedConfig

_DISCARD_KWS = {
    "inputs_embeds",
    "position_ids",
    "token_type_ids",
    "past_key_values",
    "use_cache",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
}


def patch_flamingo_for_peft(model):
    """Give OpenFlamingo just enough HF‑style interface for PEFT and preserve gradients."""

    # Store original forward
    original_forward = model.forward

    # Define the combined forward patch
    def _forward(self, *args, **kwargs):
        # ---- make PEFT happy -------------------------------------------
        if "input_ids" in kwargs and "lang_x" not in kwargs:
            kwargs["lang_x"] = kwargs.pop("input_ids")

        # Remove unwanted kwargs
        for k in list(kwargs.keys()):
            if k in _DISCARD_KWS:
                kwargs.pop(k)

        # Handle return_loss flag
        return_loss = kwargs.pop("return_loss", False)

        # ---- real forward ----------------------------------------------
        out = original_forward(*args, **kwargs)

        # ---- remove the .detach() --------------------------------------
        if isinstance(out, tuple):
            # Handle tuple output (logits,) or (loss, logits)
            if len(out) == 1:
                logits = out[0]
                loss = None
            else:
                loss, logits = out
        else:
            # Handle FlamingoOutput
            logits = out.logits
            loss = out.loss

        # Return in same format as input
        if not isinstance(out, tuple):
            return type(out)(loss=loss, logits=logits)
        return (loss, logits) if loss is not None else (logits,)

    # Apply the forward patch
    model.forward = MethodType(_forward, model)

    # Add PEFT interface methods
    def _prep(self, input_ids=None, **kwargs):
        return {"lang_x": input_ids, **kwargs}

    model.prepare_inputs_for_generation = MethodType(_prep, model)

    # Add minimal config
    if not hasattr(model, "config"):
        model.config = PretrainedConfig()
    model.config.model_type = "openflamingo"
    model.config.is_encoder_decoder = False

    # Add embedding accessors
    if not hasattr(model, "get_input_embeddings"):

        def _get(self):
            return self.lang_encoder.get_input_embeddings()

        def _set(self, new_emb):
            self.lang_encoder.set_input_embeddings(new_emb)

        model.get_input_embeddings = MethodType(_get, model)
        model.set_input_embeddings = MethodType(_set, model)

    return model


# ------------------------------------------------------------------
# ── MAIN ───────────────────────────────────────────────────────────
# ------------------------------------------------------------------
def generate_caption(
    model, tokenizer, image_processor, image_path, device=None, max_tokens=30
):
    """Generate a caption using OpenFlamingo's custom generation approach"""
    try:
        # Get device from model if not provided
        if device is None:
            device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Process image
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_processor(image)
        vision_x = image_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)

        # Use OpenFlamingo's native generate method if available
        prompt = "<image><|endofchunk|>"
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)

        # Try to use the native generate method first
        try:
            outputs = model.generate(
                vision_x=vision_x,
                lang_x=prompt_tokens.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
            )

            # Extract the generated text
            generated_ids = outputs[0][
                prompt_tokens.input_ids.shape[1] :
            ]  # Skip prompt tokens
            generated_caption = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            return generated_caption

        except (AttributeError, TypeError) as e:
            logger.warning(
                f"Native generate failed, falling back to manual generation: {e}"
            )

            # Fallback to manual token-by-token generation
            current_input_ids = prompt_tokens.input_ids
            current_attention_mask = prompt_tokens.attention_mask

            # Generate tokens one by one
            for _ in range(max_tokens):
                outputs = model(
                    vision_x=vision_x,
                    lang_x=current_input_ids,
                    attention_mask=current_attention_mask,
                )

                # Get next token probabilities
                next_token_logits = outputs[0][
                    :, -1, :
                ]  # Assuming outputs is logits or (loss, logits)

                # Apply temperature and top-p sampling
                probs = F.softmax(next_token_logits / 0.8, dim=-1)

                # Top-p sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > 0.9
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs = probs.masked_fill(indices_to_remove, 0.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to input
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                current_attention_mask = torch.ones_like(current_input_ids)

                # Stop if we generate EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break

            # Extract generated text (excluding prompt)
            generated_ids = current_input_ids[0][prompt_tokens.input_ids.shape[1] :]
            generated_caption = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            return generated_caption

    except Exception as e:
        logger.error(f"Error generating caption: {e}")
        return "Error generating caption"


class COCODataset(Dataset):
    def __init__(self, json_path, img_dir, proc, tok, max_len=100):
        self.proc, self.tok, self.max_len = proc, tok, max_len
        self.img_dir = img_dir

        # Load JSON data
        with open(json_path) as f:
            data = json.load(f)

        # Create samples directly from the data
        self.samples = []

        # Handle both list format and dict format
        if isinstance(data, list):
            # Your dataset is already a list of samples
            self.samples = data
        elif isinstance(data, dict) and "annotations" in data:
            # Handle COCO format if needed
            self.samples = data["annotations"]
        else:
            logger.error(f"Unexpected data format: {type(data)}")
            raise ValueError("Unsupported dataset format")

        # Log dataset statistics
        total_images = len(self.samples)
        logger.info(f"Dataset statistics:")
        logger.info(f"- Total images/captions: {total_images}")
        logger.info(f"- Format: Single caption per image")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Extract image ID and path
        img_id = item.get("image_id", None)

        # Handle different possible image path formats
        if "image_path" in item:
            # Direct path in the dataset
            img_path = item["image_path"]
        elif isinstance(img_id, int):
            # COCO-style numerical ID
            img_path = os.path.join(self.img_dir, f"COCO_train2014_{img_id:012d}.jpg")
        else:
            # Fallback - use ID as path
            img_path = os.path.join(self.img_dir, str(img_id))

        # Process image
        try:
            image = Image.open(img_path).convert("RGB")
            v = self.proc(image)
            v = v.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            # Return a placeholder or skip
            raise e

        # Get caption, checking both possible fields
        caption = item.get("caption", item.get("original_caption", ""))

        # Process text
        prompt = "<image><|endofchunk|>"
        text = f"{prompt} {caption}"

        try:
            enc = self.tok(
                text,
                return_tensors="pt",
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
            )

            labels = enc.input_ids.clone().squeeze(0)
            p_len = len(self.tok(prompt, add_special_tokens=True).input_ids)
            labels[:p_len] = -100  # Mask prompt tokens
            labels[labels == self.tok.pad_token_id] = -100  # Mask padding
        except Exception as e:
            logger.error(f"Error tokenizing text '{text}': {e}")
            raise e

        return {
            "vision_x": v,
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": labels,
            "image_path": img_path,
            "caption": caption,
            "image_id": img_id,
        }


def configure_lora_for_flamingo(model):
    """Configure LoRA target modules for OpenFlamingo"""
    # Extract target module types instead of full paths
    target_modules_types = set()

    # First pass: identify module types to target
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if "lang_encoder" in name:
                if "query_key_value" in name or any(
                    x in name for x in ["to_q", "to_kv", "to_out"]
                ):
                    # Extract just the last part of the name as the type
                    parts = name.split(".")
                    for i in range(len(parts) - 1, -1, -1):
                        if parts[i] in ["query_key_value", "to_q", "to_kv", "to_out"]:
                            target_modules_types.add(parts[i])
                            break

    # Log discovered module types
    target_modules = list(target_modules_types)
    logger.info(f"Discovered LoRA target module types: {target_modules}")

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )

    return lora_config


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_model(model_size="medium"):
    """Load OpenFlamingo model, image processor, and tokenizer."""
    logger.info(f"Loading OpenFlamingo {model_size} model...")

    # Load model config and checkpoint first
    if model_size == "medium":
        model_path = "openflamingo/OpenFlamingo-4B-vitl-rpj3b"
        logger.info("Loaded ViT-L-14 model config.")
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

    # Download and load checkpoint first
    logger.info(f"Loading checkpoint from: {model_path}")
    checkpoint_path = hf_hub_download(model_path, "checkpoint.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Initialize model with proper CLIP weights
    from open_flamingo import create_model_and_transforms

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",  # Use OpenAI CLIP weights
        lang_encoder_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
        tokenizer_path="togethercomputer/RedPajama-INCITE-Base-3B-v1",
        cross_attn_every_n_layers=1,
    )

    # Load state dict with strict=False to handle any minor mismatches
    model.load_state_dict(checkpoint, strict=False)
    logger.info("Successfully loaded checkpoint")

    # Unfreeze the gating scalar to allow proper vision-language interaction
    if hasattr(model, "lang_to_vit_attention_gating"):
        model.lang_to_vit_attention_gating.requires_grad = True
        logger.info("Unfroze vision-language gating scalar")

    # Configure tokenizer
    logger.info("Configuring tokenizer with special tokens...")
    tokenizer.pad_token = "<PAD>"
    tokenizer.pad_token_id = 50279
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 0
    tokenizer.bos_token = "<|endoftext|>"
    tokenizer.bos_token_id = 0

    # Add custom tokens
    special_tokens = {
        "additional_special_tokens": ["<|endofchunk|>", "<image>"],
        "pad_token": "<PAD>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    }
    tokenizer.add_special_tokens(special_tokens)

    # Log token configuration
    logger.info("Token configuration:")
    logger.info(f"  PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"  EOS: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"  BOS: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    logger.info(
        f"  <|endofchunk|>: ID: {tokenizer.convert_tokens_to_ids('<|endofchunk|>')}"
    )

    return model, image_processor, tokenizer


def evaluate_model(model, val_dl, tokenizer, device, num_samples=100):
    """Comprehensive evaluation of the model"""
    model.eval()
    all_generated = []
    all_references = []
    total_loss = 0
    num_batches = 0

    # Get model's dtype
    dtype = next(model.parameters()).dtype
    logger.info(f"Using dtype: {dtype} for evaluation")

    # Get image processor from model
    image_processor = model.vision_encoder.image_processor
    logger.info("Using model's built-in image processor for evaluation")

    # Metrics
    bleu_scores = []
    exact_matches = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_dl:
            # Forward pass with correct dtype
            outputs = model(
                vision_x=batch["vision_x"].to(device, dtype=dtype),
                lang_x=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )

            # Calculate loss
            total_loss += outputs.loss.item()
            num_batches += 1

            # Generate captions
            for i in range(min(len(batch["image_path"]), num_samples - total_samples)):
                try:
                    generated = generate_caption(
                        model,
                        tokenizer,
                        image_processor,
                        batch["image_path"][i],
                        device,
                    )
                    reference = batch["caption"][i]

                    # Store for BLEU calculation
                    all_generated.append(word_tokenize(generated.lower()))
                    all_references.append([word_tokenize(reference.lower())])

                    # Calculate exact matches
                    if generated.lower() == reference.lower():
                        exact_matches += 1
                    total_samples += 1

                except Exception as e:
                    logger.error(f"Error generating caption: {e}")
                    continue

                if total_samples >= num_samples:
                    break

            if total_samples >= num_samples:
                break

    # Calculate metrics
    avg_loss = total_loss / num_batches
    exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0

    # Calculate BLEU scores
    bleu_1 = corpus_bleu(all_references, all_generated, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(all_references, all_generated, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(all_references, all_generated, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(
        all_references, all_generated, weights=(0.25, 0.25, 0.25, 0.25)
    )

    return {
        "loss": avg_loss,
        "exact_match_rate": exact_match_rate,
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "generated_samples": list(zip(all_generated, all_references)),
    }


def main() -> None:
    # ----------  runtime / hardware setup ----------
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Adjusted hyperparameters
    batch_size = 32
    epochs = 10
    lr = 1e-4
    grad_acc = 2
    max_len = 100
    warmup_steps = 200
    max_grad_norm = 1.0
    validate_every = 1
    patience = 3

    out_dir = "./flamingo_lora_output"
    os.makedirs(out_dir, exist_ok=True)

    # ----------  distilled dataset paths ----------
    captions_json = "./vlm_distillation_output/distilled_dataset.json"
    images_dir = "./images"

    # ----------  load model and dataset ----------
    try:
        logger.info("Loading OpenFlamingo model...")
        model, img_proc, tok = load_model("medium")
        model = patch_flamingo_for_peft(model)
        logger.info("Model loaded and patched successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return

    # ----------  configure LoRA ----------
    try:
        logger.info("Configuring LoRA for OpenFlamingo...")

        # Discover target module types
        target_modules_types = set()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if "lang_encoder" in name:
                    if "query_key_value" in name or any(
                        x in name for x in ["to_q", "to_kv", "to_out"]
                    ):
                        # Extract just the last part of the name as the type
                        parts = name.split(".")
                        for i in range(len(parts) - 1, -1, -1):
                            if parts[i] in [
                                "query_key_value",
                                "to_q",
                                "to_kv",
                                "to_out",
                            ]:
                                target_modules_types.add(parts[i])
                                break

        target_modules = list(target_modules_types)
        logger.info(f"Discovered LoRA target module types: {target_modules}")

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_cfg)
        model.to(device, dtype)

        # Warm-start the gating scalar
        if hasattr(model, "lang_to_vit_attention_gating"):
            model.lang_to_vit_attention_gating.data[:] = 1.0
            logger.info("Warm-started vision-language gating scalar to 1.0")
    except Exception as e:
        logger.error(f"Error configuring LoRA: {e}")
        logger.error(traceback.format_exc())
        return

    # Log model info
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized:")
    logger.info(f"- Total parameters: {total:,}")
    logger.info(f"- Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    logger.info(f"- Device: {next(model.parameters()).device}")
    logger.info(f"- Dtype: {next(model.parameters()).dtype}")

    # ----------  datasets & dataloaders ----------
    try:
        logger.info("Loading dataset...")
        full_ds = COCODataset(captions_json, images_dir, img_proc, tok, max_len=max_len)
        n = len(full_ds)

        if n == 0:
            logger.error("Dataset is empty! Check dataset path and format.")
            return

        train_ds, val_ds, _ = torch.utils.data.random_split(
            full_ds,
            [int(0.8 * n), int(0.1 * n), n - int(0.9 * n)],
            generator=torch.Generator().manual_seed(42),
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        logger.info(
            f"Created dataloaders - Train: {len(train_dl)} batches, Val: {len(val_dl)} batches"
        )
    except Exception as e:
        logger.error(f"Error creating dataset/dataloaders: {e}")
        logger.error(traceback.format_exc())
        return

    # ----------  optimiser & scheduler ----------
    try:
        logger.info("Setting up optimizer and scheduler...")
        optim = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        opt_params = sum(p.numel() for p in optim.param_groups[0]["params"])
        logger.info(f"Optimizer sees {opt_params:,} parameters")

        # Calculate total steps with gradient accumulation
        tot_steps = (len(train_dl) // grad_acc) * epochs
        sched = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps, num_training_steps=tot_steps
        )
    except Exception as e:
        logger.error(f"Error setting up optimizer/scheduler: {e}")
        logger.error(traceback.format_exc())
        return

    # ----------  training loop ----------
    global_step = 0
    best_val_loss = float("inf")
    no_improvement = 0
    train_epoch_losses, val_epoch_losses = [], []
    best_metrics = None

    for epoch in range(1, epochs + 1):
        logger.info(f"Starting epoch {epoch}/{epochs}")
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")

        # Initialize gradients only once per gradient accumulation step
        optim.zero_grad()

        for step, batch in enumerate(pbar, 1):
            try:
                # Log shapes for debugging in first step of first epoch
                if step == 1 and epoch == 1:
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"{k} shape: {v.shape}")

                # Move tensors to device and cast vision input to correct dtype
                vis = batch["vision_x"].to(device, dtype)
                ids = batch["input_ids"].to(device)
                att = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # Forward pass
                out = model(
                    vision_x=vis,
                    lang_x=ids,  # Use lang_x consistently
                    attention_mask=att,
                    labels=labels,
                )

                # Get loss - handle different output formats
                if isinstance(out, tuple):
                    loss = out[0]
                else:
                    loss = out.loss

                # Normalize loss for gradient accumulation
                loss = loss / grad_acc
                loss.backward()

                # Accumulate loss for logging (un-normalized)
                running += loss.item() * grad_acc

                # Update weights according to accumulation schedule
                if step % grad_acc == 0 or step == len(train_dl):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # Check for NaN gradients
                    if any(
                        p.grad is not None and torch.isnan(p.grad).any()
                        for p in model.parameters()
                    ):
                        logger.warning(
                            f"NaN gradients detected at step {step}! Skipping update."
                        )
                        # Zero out gradients and continue
                        optim.zero_grad()
                        continue

                    # Perform update
                    optim.step()
                    sched.step()
                    optim.zero_grad()
                    global_step += 1

                # Update progress bar with metrics
                pbar.set_postfix(
                    {
                        "loss": f"{running/step:.4f}",
                        "lr": f"{sched.get_last_lr()[0]:.2e}",
                    }
                )

            except Exception as e:
                logger.error(f"Error in training step {step}: {e}")
                logger.error(traceback.format_exc())
                # Try to continue with next batch
                continue

        # Record epoch loss
        epoch_train_loss = running / len(train_dl)
        train_epoch_losses.append(epoch_train_loss)
        logger.info(f"Epoch {epoch} training loss: {epoch_train_loss:.4f}")

        # Validation with comprehensive evaluation
        if epoch % validate_every == 0:
            logger.info("Running validation...")
            try:
                metrics = evaluate_model(model, val_dl, tok, device)

                # Log metrics
                logger.info(f"Epoch {epoch} validation metrics:")
                logger.info(f"- Loss: {metrics['loss']:.4f}")
                logger.info(f"- Exact Match Rate: {metrics['exact_match_rate']:.4f}")
                logger.info(f"- BLEU-1: {metrics['bleu_1']:.4f}")
                logger.info(f"- BLEU-2: {metrics['bleu_2']:.4f}")
                logger.info(f"- BLEU-3: {metrics['bleu_3']:.4f}")
                logger.info(f"- BLEU-4: {metrics['bleu_4']:.4f}")

                # Log some sample generations
                logger.info("\nSample generations:")
                for gen, ref in metrics["generated_samples"][:3]:
                    logger.info(f"Generated: {' '.join(gen)}")
                    logger.info(f"Reference: {' '.join(ref[0])}")
                    logger.info("-" * 50)

                # Early stopping check using BLEU-4 score
                if metrics["bleu_4"] > best_val_loss:
                    best_val_loss = metrics["bleu_4"]
                    best_metrics = metrics

                    # Save best model
                    try:
                        logger.info(
                            f"Saving new best model (BLEU-4 = {metrics['bleu_4']:.4f})..."
                        )
                        best_dir = f"{out_dir}/best_model"
                        os.makedirs(best_dir, exist_ok=True)
                        model.save_pretrained(best_dir)
                        # Save tokenizer too if needed
                        tok.save_pretrained(best_dir)
                    except Exception as e:
                        logger.error(f"Error saving best model: {e}")

                    no_improvement = 0
                else:
                    no_improvement += 1
                    logger.info(
                        f"No improvement for {no_improvement} validation checks"
                    )

                if no_improvement >= patience:
                    logger.info(
                        f"Early stopping after {patience} epochs without improvement"
                    )
                    break

                val_epoch_losses.append(metrics["loss"])

            except Exception as e:
                logger.error(f"Error during validation: {e}")
                logger.error(traceback.format_exc())
                # Continue training even if validation fails
                continue

        # Save checkpoint each epoch
        try:
            logger.info(f"Saving checkpoint for epoch {epoch}...")
            checkpoint_dir = f"{out_dir}/checkpoint_epoch_{epoch}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

        # Monitor GPU usage after clearing cache
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

        gc.collect()

    # Save final model and best metrics
    try:
        logger.info("Saving final model...")
        model.save_pretrained(f"{out_dir}/final_model")

        if best_metrics:
            with open(f"{out_dir}/best_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)
        logger.info("✓ Training finished and model saved")
    except Exception as e:
        logger.error(f"Error saving final results: {e}")

    # Plot training curves
    try:
        min_len = min(len(train_epoch_losses), len(val_epoch_losses))
        if min_len > 0:
            epochs_range = range(1, min_len + 1)

            plt.figure(figsize=(8, 5))
            plt.plot(epochs_range, train_epoch_losses[:min_len], label="Train Loss")
            plt.plot(epochs_range, val_epoch_losses[:min_len], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-entropy Loss")
            plt.title("OpenFlamingo LoRA Fine-tune — Loss per Epoch")
            plt.legend()
            plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/training_curves.png")
            plt.close()
            logger.info(f"Training curves saved to {out_dir}/training_curves.png")
    except Exception as e:
        logger.error(f"Error generating training curves: {e}")


# run it
if __name__ == "__main__":
    logger.info(f"Starting run at {datetime.now()}")
    main()
