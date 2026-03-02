"""
Weights & Biases Logger for IDU Training with FlowEdit
Usage: Import and call init_wandb() at the start of training
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def init_wandb(
    project: str = "skyfall-gs-idu",
    entity: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    notes: Optional[str] = None,
    resume: Optional[str] = None,
    mode: str = "online"  # or "offline" for local logging
):
    """
    Initialize Weights & Biases logging for IDU training

    Args:
        project: W&B project name
        entity: W&B username or team name
        name: Run name (auto-generated if None)
        config: Dictionary of hyperparameters
        tags: List of tags for this run
        notes: Description of this run
        resume: Resume from a previous run ID
        mode: "online", "offline", or "disabled"

    Returns:
        wandb.run object or None if wandb is not available
    """
    if not WANDB_AVAILABLE:
        return None

    # Auto-generate run name with timestamp
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"idu-flowedit-{timestamp}"

    # Default tags
    if tags is None:
        tags = ["idu-training", "flowedit", "sd3", "12gb-vram"]

    # Default config with FlowEdit fixes
    default_config = {
        # Training
        "total_iterations": 10000,
        "resolution": 1,
        "sh_degree": 1,

        # IDU
        "idu_enabled": True,
        "idu_num_cams": 1,
        "idu_train_ratio": 0.75,

        # FlowEdit (CRITICAL FIXES)
        "flowedit_enabled": True,
        "flowedit_model": "SD3",
        "flowedit_t5_device": "cpu",  # CRITICAL: T5 on CPU
        "flowedit_transformer_quantization": "4bit",
        "idu_render_size": 256,
        "idu_flow_edit_n_min": 4,
        "idu_flow_edit_n_max": 10,

        # Densification (CRITICAL FOR MEMORY)
        "densify_grad_threshold": 0.0005,  # Increased from 0.0002
        "idu_densify_until_iter": 9000,

        # Loss weights
        "lambda_depth": 0,
        "lambda_pseudo_depth": 0.5,
        "lambda_opacity": 0,

        # System
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    }

    # Merge with user config
    if config is not None:
        default_config.update(config)

    # Initialize wandb
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=default_config,
        tags=tags,
        notes=notes,
        resume=resume,
        mode=mode,
    )

    # Log system info
    wandb.config.update({
        "hostname": os.uname().nodename,
        "cuda_available": True,  # Assume CUDA is available
    })

    return run


def log_metrics(metrics: Dict[str, float], step: int, commit: bool = True):
    """
    Log metrics to wandb

    Args:
        metrics: Dictionary of metric names and values
        step: Current iteration/step number
        commit: Whether to commit the log (set False for multiple log calls per step)
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    wandb.log(metrics, step=step, commit=commit)


def log_flowedit_images(images: list, captions: list, step: int):
    """
    Log FlowEdit refined images to wandb

    Args:
        images: List of PIL images or numpy arrays
        captions: List of captions for each image
        step: Current iteration/step number
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    wandb_images = [wandb.Image(img, caption=cap) for img, cap in zip(images, captions)]
    wandb.log({"flowedit_refined": wandb_images}, step=step)


def finish_wandb():
    """Finish the wandb run"""
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


# Example usage in training script
if __name__ == "__main__":
    # Example: Initialize wandb
    config = {
        "dataset_name": "JAX_068",
        "start_checkpoint": "chkpnt30000.pth",
    }

    run = init_wandb(
        project="skyfall-gs-idu",
        name="test-run",
        config=config,
        tags=["test"],
        notes="Test run for wandb integration",
        mode="offline"  # Use offline mode for testing
    )

    # Example: Log metrics during training
    for i in range(10):
        metrics = {
            "loss": 0.1 * (10 - i),
            "psnr": 18.0 + i * 0.5,
            "num_gaussians": 500000 + i * 10000,
        }
        log_metrics(metrics, step=i)

    # Finish
    finish_wandb()
    print("Wandb test completed!")
