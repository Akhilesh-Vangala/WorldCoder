# Zero-Shot WorldCoder: Implementation Guide

## Overview

Pure zero-shot approach: **Frozen V-JEPA + LLM + Physics Verifier**

No dataset generation needed!

## Architecture

```
Start Video → V-JEPA (frozen) → Embeddings → LLM → Blender Code → Physics Verifier → Refined Code
```

## Setup

### 1. Install Dependencies

```bash
pip install torch numpy opencv-python openai
```

### 2. Set API Keys

```bash
export OPENAI_API_KEY="your-key-here"
# Or use Claude, Gemini, etc.
```

### 3. Load V-JEPA (Optional)

If you have V-JEPA model:
- Update `VJEPAEncoder._load_model()` with actual loading code
- Placeholder embeddings work for testing

## Usage

### Basic Example

```python
from zero_shot_worldcoder import ZeroShotWorldCoder
import numpy as np

# Initialize
coder = ZeroShotWorldCoder(
    llm_api_key="your-key",  # Optional
    max_iterations=3
)

# Load videos (any source!)
start_video = load_video("start.mp4")  # Your video loading function
goal_video = load_video("goal.mp4")

# Transform
code, scores = coder.transform(start_video, goal_video)

# Generated code is ready to execute in Blender!
print(code)
```

### Using Your Generated Blender Scenes

```python
from example_usage import example_use_existing_blend_files

# Uses dataset/blender_files/*.blend
code, scores = example_use_existing_blend_files()
```

## How It Works

### Step 1: Embedding Extraction
- Frozen V-JEPA extracts temporal embeddings
- Captures motion patterns (no training)

### Step 2: LLM Code Generation
- Analyzes embeddings → interprets motion
- Generates Blender code with physics parameters
- Uses few-shot prompting (3-5 hand-crafted examples)

### Step 3: Physics Verification
- Executes code in Blender
- Renders result video
- Compares with goal video
- Checks physics validity

### Step 4: Iterative Refinement
- If validation fails, generates feedback
- LLM refines code based on feedback
- Repeats until valid or max iterations

## Components

### VJEPAEncoder
- Extracts temporal embeddings from videos
- Frozen (no training)
- Placeholder available if model not loaded

### LLMCodeGenerator
- Generates Blender code from embeddings
- Uses few-shot prompting
- Supports OpenAI, Claude (can extend)

### PhysicsVerifier
- Executes Blender code
- Renders and evaluates result
- Provides feedback for refinement

## Customization

### Add Few-Shot Examples

Edit `LLMCodeGenerator._few_shot_examples()`:
```python
def _few_shot_examples(self) -> str:
    return """
    Your custom examples here...
    """
```

### Change LLM Provider

Modify `LLMCodeGenerator.generate_code()`:
- Switch OpenAI → Claude
- Use local LLM (Ollama, etc.)
- Custom API

### Improve Physics Verification

Enhance `PhysicsVerifier._evaluate()`:
- Add LPIPS for visual similarity
- Implement optical flow for TFA
- Add collision detection for PVS

## Advantages

✅ **No dataset generation** - Use any videos
✅ **Fast iteration** - Test immediately
✅ **Self-correcting** - Physics verifier ensures correctness
✅ **General purpose** - Works on any temporal transformation

## Next Steps

1. **Test with your 5 generated scenes**
   ```python
   python example_usage.py
   ```

2. **Add real V-JEPA model** (when available)
   - Update `VJEPAEncoder._load_model()`

3. **Customize few-shot examples** for your domain

4. **Improve physics verifier** with better metrics

5. **Test on diverse videos** (any source!)

## Troubleshooting

**V-JEPA not loaded?**
- Placeholder works for testing
- Add real model when available

**LLM errors?**
- Check API key
- Verify internet connection
- Try different model (gpt-3.5-turbo is cheaper)

**Blender execution fails?**
- Check Blender path in `PhysicsVerifier`
- Verify code syntax
- Check Blender version compatibility

## For Publication

This approach demonstrates:
- ✅ Zero-shot temporal video understanding
- ✅ Physics-aware code generation
- ✅ Self-correcting refinement loop
- ✅ No dataset requirements

Perfect for CVPR if it works well on diverse examples!





