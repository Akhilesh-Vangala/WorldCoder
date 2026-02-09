# Results Generation Guide for CVPR Paper

This guide explains how to generate comprehensive results for your CVPR paper.

## Quick Start

### 1. Generate Main Results

Run the comprehensive evaluation on all test pairs:

```bash
python generate_cvpr_results.py
```

This will:
- Evaluate all 5 test pairs
- Generate Blender code for each pair
- Compute all metrics (PFF, TFA, PVS)
- Compare with ground truth physics parameters
- Save results to `cvpr_results/` directory

**Output files:**
- `cvpr_results/all_results.json` - Complete results
- `cvpr_results/summary_report.txt` - Human-readable summary
- `cvpr_results/results_table.tex` - LaTeX table for paper
- `cvpr_results/generated_code_pair_XXXX.py` - Generated code for each pair

### 2. Analyze Results

Generate visualizations and insights:

```bash
python analyze_results.py
```

**Requirements:** 
- `matplotlib` (optional, for plots)
- `pandas` (optional, for CSV export)

**Output files:**
- `cvpr_results/metrics_plot.png` - Metrics visualization
- `cvpr_results/parameter_comparison.png` - Parameter accuracy plot
- `cvpr_results/results_table.csv` - CSV table
- `cvpr_results/insights.txt` - Key insights

### 3. Run Ablation Studies

Compare different configurations:

```bash
python batch_evaluation.py
```

**Configurations tested:**
- `baseline` - Full pipeline (3 iterations, V-JEPA, physics verifier)
- `no_iterations` - Single iteration (no refinement)
- `no_physics_verifier` - Without physics verification
- `no_vjepa` - Without V-JEPA embeddings (placeholder)

**Output files:**
- `cvpr_results/ablation_studies/*_results.json` - Results per configuration
- `cvpr_results/ablation_studies/ablation_comparison.json` - Comparison
- `cvpr_results/ablation_studies/ablation_table.tex` - LaTeX table

## Results Structure

### Main Results (`all_results.json`)

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "num_pairs": 5,
  "results": [
    {
      "pair_id": 1,
      "scores": {
        "per_frame_fidelity": 0.85,
        "temporal_flow_alignment": 0.82,
        "physics_validity_score": 0.91
      },
      "predicted_physics": {...},
      "ground_truth_physics": {...},
      "parameter_accuracy": {...}
    }
  ],
  "statistics": {
    "per_frame_fidelity": {
      "mean": 0.85,
      "std": 0.05,
      "min": 0.80,
      "max": 0.90
    }
  }
}
```

### Metrics Explained

1. **Per-Frame Fidelity (PFF)**: Visual similarity between generated and goal videos
   - Computed using PSNR between frames
   - Range: [0, 1]
   - Success threshold: > 0.85

2. **Temporal Flow Alignment (TFA)**: Motion consistency
   - Computed using optical flow comparison
   - Range: [0, 1]
   - Success threshold: > 0.80

3. **Physics Validity Score (PVS)**: Physical plausibility
   - Checks for physics violations (static objects, teleportation, etc.)
   - Range: [0, 1]
   - Success threshold: > 0.90

## For Paper Writing

### Main Results Table

Use `cvpr_results/results_table.tex` directly in your LaTeX document:

```latex
\input{cvpr_results/results_table.tex}
```

### Figures

1. **Metrics Plot**: `cvpr_results/metrics_plot.png`
   - Shows all three metrics across test pairs
   - Ready for inclusion in paper

2. **Parameter Comparison**: `cvpr_results/parameter_comparison.png`
   - Scatter plots of predicted vs ground truth parameters
   - Shows prediction accuracy

### Statistics

Reference values from `cvpr_results/summary_report.txt`:

```
Per-Frame Fidelity: 0.85 ± 0.05
Temporal Flow Alignment: 0.82 ± 0.04
Physics Validity Score: 0.91 ± 0.03
```

### Ablation Study

Use `cvpr_results/ablation_studies/ablation_table.tex` for ablation results.

## Customization

### Change Number of Test Pairs

Edit `NUM_TEST_PAIRS` in `generate_cvpr_results.py`:

```python
NUM_TEST_PAIRS = 5  # Change this
```

### Add New Ablation Configuration

Edit `ABLATION_CONFIGS` in `batch_evaluation.py`:

```python
ABLATION_CONFIGS = {
    'my_config': {
        'max_iterations': 2,
        'llm_provider': 'gemini',
        'llm_model': 'gemini-2.0-flash-exp',
        'use_vjepa': True,
        'use_physics_verifier': True,
    },
}
```

### Change Evaluation Metrics

Modify the `_evaluate` method in `zero_shot_worldcoder.py` to add new metrics.

## Troubleshooting

### Missing Dependencies

```bash
pip install matplotlib pandas  # For analysis script
```

### Blender Not Found

Set `BLENDER_PATH` in the scripts:

```python
BLENDER_PATH = '/path/to/blender'
```

### API Key Issues

Set environment variable:

```bash
export GEMINI_API_KEY="your-key-here"
```

Or edit the scripts directly.

### Out of Memory

Reduce number of frames or test pairs:

```python
num_frames = 20  # Instead of 30
NUM_TEST_PAIRS = 3  # Instead of 5
```

## Expected Runtime

- **Single pair evaluation**: ~5-10 minutes
- **Full evaluation (5 pairs)**: ~30-50 minutes
- **Ablation studies (4 configs × 5 pairs)**: ~2-3 hours

## Next Steps

1. ✅ Run `generate_cvpr_results.py` to get main results
2. ✅ Run `analyze_results.py` to generate visualizations
3. ✅ Run `batch_evaluation.py` for ablation studies
4. ✅ Review `cvpr_results/` directory for all outputs
5. ✅ Use LaTeX tables and figures in your paper

## Questions?

Check the code comments or the main README.md for more details.

