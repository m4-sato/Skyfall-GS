"""
Example: How to integrate wandb into train.py
Add these snippets at the appropriate locations
"""

# ============================================================
# 1. Add import at the top of train.py (after other imports)
# ============================================================
from wandb_logger import init_wandb, log_metrics, log_flowedit_images, finish_wandb

# ============================================================
# 2. Initialize wandb after argument parsing
#    (around line 400-500 in train.py, after args are defined)
# ============================================================
def training_idu(dataset, opt, pipe, start_checkpoint_path=None):
    # ... existing code ...

    # Initialize wandb
    wandb_run = init_wandb(
        project="skyfall-gs-idu",
        entity=None,  # Set your wandb username here
        name=f"idu-{dataset.model_path.split('/')[-1]}",
        config={
            # Dataset info
            "dataset_path": dataset.source_path,
            "dataset_type": dataset.datasets_type,
            "num_train_images": len(scene.getTrainCameras()),
            "num_test_images": len(scene.getTestCameras()),

            # Training params
            "iterations": opt.iterations,
            "position_lr_init": opt.position_lr_init,
            "feature_lr": opt.feature_lr,
            "opacity_lr": opt.opacity_lr,
            "scaling_lr": opt.scaling_lr,
            "rotation_lr": opt.rotation_lr,

            # IDU params
            "idu_num_cams": opt.idu_num_cams,
            "idu_train_ratio": opt.idu_train_ratio,
            "idu_episode_iterations": opt.idu_episode_iterations,
            "densify_grad_threshold": opt.densify_grad_threshold,
            "idu_densify_until_iter": opt.idu_densify_until_iter,

            # FlowEdit params (CRITICAL)
            "idu_use_flow_edit": opt.idu_use_flow_edit,
            "idu_refine": opt.idu_refine,
            "idu_render_size": opt.idu_render_size,
            "idu_flow_edit_n_min": opt.idu_flow_edit_n_min,
            "idu_flow_edit_n_max": opt.idu_flow_edit_n_max,
            "flowedit_t5_device": "cpu",  # Document the fix
            "flowedit_transformer_quantization": "4bit",

            # Loss weights
            "lambda_depth": opt.lambda_depth,
            "lambda_pseudo_depth": opt.lambda_pseudo_depth,
            "lambda_opacity": opt.lambda_opacity,

            # System
            "checkpoint": start_checkpoint_path,
        },
        tags=["idu-training", "flowedit", dataset.datasets_type, "12gb-vram"],
        notes=f"IDU training with FlowEdit on {dataset.datasets_type}. T5 CPU offload fix applied. Densify threshold: {opt.densify_grad_threshold}"
    )

    # ... rest of the function ...


# ============================================================
# 3. Log metrics during training loop
#    (in training_idu_episode function, around line 800-900)
# ============================================================
def training_idu_episode(...):
    # ... existing training loop ...

    for iteration in range(first_iter, opt.iterations + 1):
        # ... existing training code ...

        # Compute loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss = loss + depth_loss_value + opacity_loss_value

        # WANDB: Log training metrics every 10 iterations
        if iteration % 10 == 0:
            metrics = {
                "train/loss": loss.item(),
                "train/l1_loss": Ll1.item(),
                "train/ssim": ssim_value.item(),
            }

            if depth_loss_value is not None:
                metrics["train/depth_loss"] = depth_loss_value.item()

            if opacity_loss_value is not None:
                metrics["train/opacity_loss"] = opacity_loss_value.item()

            # Add system metrics
            metrics["system/num_gaussians"] = len(gaussians.get_xyz)

            if torch.cuda.is_available():
                metrics["system/gpu_memory_mb"] = torch.cuda.memory_allocated() / 1024**2

            log_metrics(metrics, step=iteration)

        # ... rest of training loop ...


# ============================================================
# 4. Log evaluation metrics
#    (after evaluation runs, around line 900-1000)
# ============================================================
def training_idu_episode(...):
    # ... existing code ...

    # After evaluation
    if (iteration in testing_iterations):
        # ... existing evaluation code ...

        # WANDB: Log evaluation metrics
        eval_metrics = {
            "eval/psnr_test": psnr_test,
            "eval/l1_test": l1_test,
            "eval/psnr_train": psnr_train,
            "eval/l1_train": l1_train,
        }

        if psnr_train_idu is not None:
            eval_metrics["eval/psnr_train_idu"] = psnr_train_idu
            eval_metrics["eval/l1_train_idu"] = l1_train_idu

        log_metrics(eval_metrics, step=iteration)


# ============================================================
# 5. Log FlowEdit refined images
#    (in generate_idu_training_set function, after FlowEdit refinement)
# ============================================================
def generate_idu_training_set(...):
    # ... existing code ...

    if opt.idu_refine and opt.idu_use_flow_edit:
        # Run FlowEdit
        final_imgs = refine_pipe.run(
            imgs=img_np_list,
            src_prompt=src_prompt,
            tar_prompt=tar_prompt,
            T_steps=T_steps,
            n_avg=n_avg,
            src_guidance_scale=src_guidance_scale,
            tar_guidance_scale=tar_guidance_scale,
            n_min=n_min,
            n_max=n_max,
            n_max_end=n_max_end,
        )

        # WANDB: Log FlowEdit refined images
        try:
            import wandb
            if wandb.run is not None:
                log_flowedit_images(
                    images=final_imgs,
                    captions=[f"IDU_refined_{i}_elev{elevation}_rad{radius}"
                             for i in range(len(final_imgs))],
                    step=iteration  # You'll need to pass iteration as a parameter
                )
        except Exception as e:
            print(f"Warning: Could not log FlowEdit images to wandb: {e}")


# ============================================================
# 6. Log densification events
#    (when densification occurs, around line 850-900)
# ============================================================
if iteration > opt.densify_from_iter and iteration < opt.densify_until_iter:
    # ... existing densification code ...

    # WANDB: Log densification
    log_metrics({
        "densify/grad_threshold": opt.densify_grad_threshold,
        "densify/num_points_added": num_points_after - num_points_before,
        "densify/num_points_pruned": num_points_pruned,
    }, step=iteration, commit=False)


# ============================================================
# 7. Finish wandb at the end of training
#    (at the end of training_idu function)
# ============================================================
def training_idu(...):
    # ... all training code ...

    # Training complete
    print("\\n[ITERATION {}] Training complete.".format(iteration))

    # WANDB: Finish logging
    finish_wandb()

    return start_checkpoint_path


# ============================================================
# MINIMAL EXAMPLE: Add these 3 lines for basic wandb logging
# ============================================================

# 1. At the start (after imports and args)
from wandb_logger import init_wandb, log_metrics, finish_wandb
wandb_run = init_wandb(project="skyfall-gs-idu", config=vars(args))

# 2. In training loop
log_metrics({"loss": loss.item(), "psnr": psnr}, step=iteration)

# 3. At the end
finish_wandb()
