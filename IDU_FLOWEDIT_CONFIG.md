# IDU Training with FlowEdit - Configuration Documentation

## Date
2026-03-01

## Hardware Requirements
- GPU: 11.73 GiB VRAM (tested on RTX 2080 Ti / RTX 3060)
- RAM: 32GB+ recommended
- Storage: ~50GB for models and outputs

## Critical FlowEdit Fixes

### 1. T5 Text Encoder CPU Offloading
**File:** `submodules/FlowEdit/idu_refine.py`

**Changes (lines 232-247):**
```python
# T5 (text_encoder_3) を個別にロードして CPU へ
print("Loading T5 Text Encoder to CPU...")
text_encoder_3 = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder_3",  # Changed from text_encoder_2
    torch_dtype=torch.float16,
    device_map="cpu"
)

# メモリ節約のため、T5をCPUに配置したパイプラインを構築
print("Loading SD3 Pipeline...")
self.pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=transformer,
    text_encoder_3=text_encoder_3,  # Changed from text_encoder_2
    torch_dtype=torch.float16,
)
```

**Changes (lines 261-264):**
```python
# T5 (text_encoder_3) をCPUに固定するため、hookの実行デバイスをCPUに設定
if hasattr(self.pipe, 'text_encoder_3') and self.pipe.text_encoder_3 is not None:
    if hasattr(self.pipe.text_encoder_3, '_hf_hook'):
        self.pipe.text_encoder_3._hf_hook.execution_device = "cpu"
```

### 2. Prompt Embedding Device Management
**File:** `submodules/FlowEdit/FlowEdit_utils.py`

**Changes (lines 168-170):**
```python
# Move embeddings to GPU (they were encoded on CPU to save VRAM)
src_tar_prompt_embeds = src_tar_prompt_embeds.to(device)
src_tar_pooled_prompt_embeds = src_tar_pooled_prompt_embeds.to(device)
```

**Changes (lines 140-162):**
```python
# Use all three text encoders (CLIP-L, OpenCLIP-G, T5-XXL)
pipe.encode_prompt(
    prompt=src_prompt,
    prompt_2=src_prompt,  # Changed from None
    prompt_3=src_prompt,  # Changed from None
    ...
)
```

### 3. Variable Name Fix
**File:** `submodules/FlowEdit/FlowEdit_utils.py`

**Line 223:**
```python
prev_sample = prev_sample.to(Vt_tar.dtype)  # Changed from noise_pred_tar.dtype
```

## Key Training Parameters

### Memory Optimization
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Critical Parameters for 12GB VRAM
- `--densify_grad_threshold 0.0005` (increased from 0.0002 to reduce point count)
- `--idu_render_size 256` (rendering resolution for FlowEdit)
- `--idu_num_cams 1` (number of synthetic cameras per iteration)
- `--idu_num_samples_per_view 2`

### FlowEdit Parameters
- `--idu_use_flow_edit` (enable FlowEdit refinement)
- `--idu_flow_edit_n_min 4` (minimum diffusion steps)
- `--idu_flow_edit_n_max 10` (maximum diffusion steps)

### Densification Parameters
- `--idu_densify_until_iter 9000` (stop densification at 9000)
- `--idu_opacity_cooling_iterations 500` (opacity regularization period)
- `--idu_opacity_reset_interval 5000` (reset interval)

### Training Configuration
- `--idu_episode_iterations 10000` (total iterations per episode)
- `--idu_train_ratio 0.75` (75% training, 25% IDU synthetic data)
- `--lambda_pseudo_depth 0.5` (pseudo depth loss weight)
- `--lambda_depth 0` (disable regular depth loss)
- `--lambda_opacity 0` (disable opacity loss initially)

## Expected Performance

### Memory Usage
- FlowEdit inference: ~9.3 GB GPU, ~5 GB CPU (T5)
- Training peak: ~8.6 GB GPU
- Gaussian points at completion: ~770k-800k

### Timing
- FlowEdit refinement: ~25 minutes per image
- Training: ~20-25 hours for 10000 iterations

### Metrics (at iteration 7500)
- Test PSNR: 18.33
- Train PSNR: 24.21
- Train IDU PSNR: 20.05

## SD3 Text Encoder Architecture
- `text_encoder` (text_encoder_1): CLIP-ViT-L
- `text_encoder_2`: OpenCLIP-ViT/G
- `text_encoder_3`: T5-XXL (4.7B parameters) ← Must be on CPU for 12GB VRAM

## Troubleshooting

### OOM During FlowEdit
- Issue: T5 encoder trying to move to GPU
- Solution: Set `_hf_hook.execution_device = "cpu"` after `enable_model_cpu_offload()`

### OOM During Training
- Issue: Too many Gaussian splats from aggressive densification
- Solution: Increase `--densify_grad_threshold` (0.0005 or higher)

### Device Mismatch Errors
- Issue: CPU embeddings passed to GPU transformer
- Solution: Move embeddings to GPU before transformer call

## Notes
- FlowEdit uses SD3 model, not FLUX (despite the log message)
- T5 must stay on CPU for 12GB VRAM systems
- Quantization: Transformer uses 4-bit quantization, T5 uses fp16 on CPU
