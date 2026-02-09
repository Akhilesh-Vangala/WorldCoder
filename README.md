# Zero-Shot WorldCoder: V-JEPA + LLM + Physics Verifier

Pure zero-shot approach for 4D (space-time) scene editing in Blender - **no dataset generation needed!**

## Quick Start

```python
from zero_shot_worldcoder import ZeroShotWorldCoder

coder = ZeroShotWorldCoder(max_iterations=3)
code, scores = coder.transform(start_video, goal_video)
```

## Files

### Core Implementation
- **`zero_shot_worldcoder.py`** - Main pipeline (V-JEPA + LLM + Physics Verifier)
- **`example_usage.py`** - Usage examples
- **`test_zero_shot.py`** - Test script

### Reference/Examples
- **`01_FAST_SMOOTH_ROLLING.py`** - Example Blender scene script
- **`dataset/blender_files/`** - Test scenes (5 pairs)
- **`dataset/physics/`** - Reference physics parameters

### Documentation
- **`README.md`** - This file
- **`README_ZERO_SHOT.md`** - Detailed implementation guide

## How It Works

```
Input Videos → V-JEPA (frozen) → Embeddings → LLM → Blender Code → Physics Verifier → Refined Code
```

**No training data needed** - pure zero-shot with self-correcting physics loop.

## Setup

1. Install dependencies (optional):
   ```bash
   pip install torch numpy opencv-python openai
   ```

2. Set API key (optional, for LLM):
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. Test:
   ```bash
   python test_zero_shot.py
   ```

## Usage

See `example_usage.py` for examples.

## Architecture

- **V-JEPA**: Frozen encoder for temporal embeddings (zero-shot)
- **LLM**: Code generation with few-shot prompting
- **Physics Verifier**: Self-correcting validation loop

No dataset required - works with any video input!





