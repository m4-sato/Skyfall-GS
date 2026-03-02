# Weights & Biases Setup for IDU Training

## Installation

### 1. Install wandb
```bash
# In your virtual environment
source .venv/bin/activate
pip install wandb
```

### 2. Login to wandb
```bash
wandb login
```
Enter your API key from https://wandb.ai/authorize

## Quick Start

### Option A: Use the helper script (Recommended)

Add to the beginning of `train.py`:

```python
from wandb_logger import init_wandb, log_metrics, log_flowedit_images, finish_wandb

# Initialize wandb (add after argument parsing)
wandb_run = init_wandb(
    project="skyfall-gs-idu",
    entity="your-username",  # Optional: your wandb username
    name=f"idu-{args.model_path.split('/')[-1]}",
    config={
        "dataset_path": args.source_path,
        "model_path": args.model_path,
        "checkpoint": args.start_checkpoint,
        "densify_grad_threshold": args.densify_grad_threshold,
        # Add other args as needed
    },
    tags=["idu-training", "flowedit", "jax-dataset"],
    notes="IDU training with FlowEdit refinement - T5 CPU offloading fix"
)
```

### Log metrics during training:

```python
# In your training loop (around line 800-900 in train.py)
if iteration % 100 == 0:  # Log every 100 iterations
    log_metrics({
        "loss": loss.item(),
        "depth_loss": depth_loss.item() if depth_loss else 0,
        "opacity_loss": opacity_loss.item() if opacity_loss else 0,
        "num_gaussians": len(gaussians.get_xyz),
        "learning_rate": gaussians.optimizer.param_groups[0]['lr'],
    }, step=iteration)
```

### Log evaluation metrics:

```python
# After evaluation (around line 900-1000)
if iteration in [2500, 5000, 7500, 10000]:
    log_metrics({
        "psnr_test": psnr_test,
        "l1_test": l1_test,
        "psnr_train": psnr_train,
        "l1_train": l1_train,
        "psnr_train_idu": psnr_train_idu,
        "l1_train_idu": l1_train_idu,
    }, step=iteration)
```

### Log FlowEdit refined images:

```python
# After FlowEdit refinement (in generate_idu_training_set function)
if wandb.run is not None:
    from wandb_logger import log_flowedit_images
    log_flowedit_images(
        images=refined_images,
        captions=[f"IDU_{i}" for i in range(len(refined_images))],
        step=iteration
    )
```

### Finish wandb at the end:

```python
# At the end of training
finish_wandb()
```

## Option B: Manual Integration

If you prefer to integrate wandb manually:

```python
import wandb

# Initialize
wandb.init(
    project="skyfall-gs-idu",
    config={
        # Your hyperparameters
        "densify_grad_threshold": 0.0005,
        "idu_use_flow_edit": True,
        "flowedit_t5_device": "cpu",
        # etc.
    }
)

# Log during training
wandb.log({
    "loss": loss_value,
    "psnr": psnr_value,
}, step=iteration)

# Log images
wandb.log({
    "refined_image": wandb.Image(image_array)
}, step=iteration)

# Finish
wandb.finish()
```

## Configuration File

The `wandb_config.yaml` file contains all hyperparameters and settings. Load it with:

```python
import yaml

with open("wandb_config.yaml", "r") as f:
    config = yaml.safe_load(f)

wandb.init(
    project=config["project"],
    config=config["hyperparameters"],
    tags=config["run"]["tags"],
)
```

## What to Track

### Essential Metrics
- `loss`: Total training loss
- `depth_loss`: Depth reconstruction loss
- `opacity_loss`: Opacity regularization loss
- `psnr_test`: Test set PSNR
- `psnr_train`: Training set PSNR
- `psnr_train_idu`: IDU synthetic data PSNR
- `num_gaussians`: Number of Gaussian splats

### System Metrics
- `gpu_memory_allocated`: GPU memory usage (MB)
- `iteration_time`: Time per iteration (seconds)

### FlowEdit Specific
- FlowEdit refined images (before/after)
- Pseudo camera parameters
- IDU episode number

## Running with wandb

### Online Mode (Default)
```bash
./run_idu_training.sh
```

### Offline Mode (for no internet)
```bash
export WANDB_MODE=offline
./run_idu_training.sh
```

Later sync with:
```bash
wandb sync wandb/offline-run-*
```

### Disabled Mode (no logging)
```bash
export WANDB_MODE=disabled
./run_idu_training.sh
```

## Viewing Results

After training starts, wandb will print a URL like:
```
wandb: 🚀 View run at https://wandb.ai/your-username/skyfall-gs-idu/runs/abc123
```

Open this URL to see:
- Real-time training curves
- System metrics (GPU, memory)
- Hyperparameters
- Logged images
- Model artifacts

## Best Practices

1. **Tag your runs** with meaningful tags:
   - Dataset name (e.g., "jax-068")
   - Experiment type (e.g., "flowedit", "baseline")
   - Hardware (e.g., "12gb-vram", "24gb-vram")

2. **Use descriptive names**:
   ```python
   name = f"idu-flowedit-{dataset}-{timestamp}"
   ```

3. **Log important config**:
   - All command-line arguments
   - FlowEdit fixes applied
   - System information

4. **Save artifacts**:
   ```python
   # Save model checkpoint
   wandb.save("checkpoints/model.pth")

   # Save FlowEdit images
   wandb.save("idu/*/refined_images/*.png")
   ```

5. **Add notes** explaining the run:
   ```python
   notes = "Fixed CUDA OOM with T5 CPU offload. Increased densify_grad_threshold to 0.0005."
   ```

## Comparing Runs

In the wandb UI:
1. Select multiple runs
2. Click "Compare" to see side-by-side metrics
3. Use workspace to create custom visualizations

## Example: Full Integration

See `wandb_logger.py` for a complete example with:
- Automatic initialization
- Metric logging helpers
- Image logging
- Proper cleanup

## Troubleshooting

### "wandb: ERROR Unable to connect"
- Check internet connection
- Use offline mode: `export WANDB_MODE=offline`

### "wandb: WARNING Step must always increase"
- Ensure step parameter is monotonically increasing
- Don't reset iteration counter

### Large log files
- Log less frequently (every 100 iterations instead of every 1)
- Reduce image logging frequency
- Use lower resolution for logged images

## Resources

- W&B Documentation: https://docs.wandb.ai/
- W&B Guides: https://wandb.ai/site/guides
- Example integrations: https://github.com/wandb/examples
