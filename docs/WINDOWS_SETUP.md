# Windows Setup Notes

NEXUS runs on Windows with some extra steps for the local model inference pipeline.

## PyTorch and torchao Compatibility

If you use the bundled LoRA adapter (`NEXUS_LOCAL_BACKEND=adapter`), the `bitsandbytes` and `torchao` packages may need patches on Windows with PyTorch 2.5+.

### Issue 1: `torchao` — Missing `torch.intX` types

**Symptom:** `AttributeError: module 'torch' has no attribute 'int1'`

**Fix:** In your installed `torchao/quantization/quant_primitives.py`, replace any hardcoded `torch.int1` through `torch.int7` and `torch.uint1` through `torch.uint7` dictionaries with a safe loop using `getattr`:

```python
# Replace hardcoded dicts like:
#   _SUB_BYTE_INT_BOUNDS = {torch.int1: ..., torch.int2: ..., ...}
# With:
_SUB_BYTE_INT_BOUNDS = {}
for _bits in range(1, 8):
    _signed = getattr(torch, f"int{_bits}", None)
    _unsigned = getattr(torch, f"uint{_bits}", None)
    if _signed is not None:
        _SUB_BYTE_INT_BOUNDS[_signed] = (-(2 ** (_bits - 1)), 2 ** (_bits - 1) - 1)
    if _unsigned is not None:
        _SUB_BYTE_UINT_BOUNDS[_unsigned] = (0, 2 ** _bits - 1)
```

### Issue 2: `torchao` — Missing `register_constant`

**Symptom:** `AttributeError: module 'torch.utils._pytree' has no attribute 'register_constant'`

**Fix:** In `torchao/utils.py`, wrap the `register_as_pytree_constant` call:

```python
if hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant(...)
```

### Issue 3: Triton — `Failed to find CUDA`

**Symptom:** `UserWarning: Failed to find CUDA.`

This warning is **harmless**. NEXUS bypasses Triton entirely by using standard HuggingFace `transformers` + `peft` for inference. The warning comes from the Triton package being installed as a transitive dependency but not being used.

## CUDA Acceleration (Optional)

For faster inference, install the CUDA toolkit:

1. Download [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) matching your GPU
2. Ensure `nvcc` is on your PATH
3. Restart NEXUS — the `bitsandbytes` library will automatically use GPU acceleration
