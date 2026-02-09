"""
Generate CVPR Results with CLIP-Enhanced Version
Uses convergence-based iteration (stops when errors are low)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List
import subprocess
import tempfile
import shutil

from src.zero_shot_worldcoder_enhanced import EnhancedZeroShotWorldCoder
import generate_cvpr_results as gen_script

render_blend_file = gen_script.render_blend_file
load_ground_truth_physics = gen_script.load_ground_truth_physics
extract_physics_params_from_code = gen_script.extract_physics_params_from_code
compute_parameter_accuracy = gen_script.compute_parameter_accuracy
convert_to_native_types = gen_script.convert_to_native_types

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBI52gm0JaJbMbM0xeqO9EuN86p88gIHj0")
VJEPA_MODEL_PATH = PROJECT_ROOT / 'models' / 'vjepa' / 'vitl16.pth.tar'
DATASET_DIR = PROJECT_ROOT / 'dataset'
RESULTS_DIR = PROJECT_ROOT / 'cvpr_results'
CLIP_RESULTS_DIR = RESULTS_DIR / 'clip_enhanced_results'
CLIP_RESULTS_DIR.mkdir(exist_ok=True)

NUM_TEST_PAIRS = 5
MAX_ITERATIONS = 10  # Higher max, but will stop early on convergence
CONVERGENCE_THRESHOLD = 0.85  # Stop if PFF > this


def evaluate_pair_with_clip(pair_id: int, coder: EnhancedZeroShotWorldCoder) -> Dict:
    """Evaluate a single test pair with CLIP-enhanced version"""
    print(f"\n{'='*70}")
    print(f"Evaluating Pair {pair_id} with CLIP-Enhanced Pipeline")
    print(f"{'='*70}")
    
    # Load ground truth
    ground_truth = load_ground_truth_physics(pair_id)
    
    # Load videos
    start_blend = DATASET_DIR / 'blender_files' / f'start_{pair_id:04d}.blend'
    goal_blend = DATASET_DIR / 'blender_files' / f'goal_{pair_id:04d}.blend'
    
    if not start_blend.exists() or not goal_blend.exists():
        print(f"❌ Files not found for pair {pair_id}")
        return None
    
    print(f"[Step 1] Rendering videos...")
    start_video = render_blend_file(str(start_blend), num_frames=30)
    goal_video = render_blend_file(str(goal_blend), num_frames=30)
    print(f"  ✅ Start video: {start_video.shape}, brightness: {start_video.mean():.1f}")
    print(f"  ✅ Goal video: {goal_video.shape}, brightness: {goal_video.mean():.1f}")
    
    # Run transformation with CLIP
    print(f"[Step 2] Running CLIP-enhanced transformation pipeline...")
    try:
        code, scores = coder.transform(start_video, goal_video)
        
        # Extract physics parameters from code
        predicted_params = extract_physics_params_from_code(code)
        
        # Compute parameter accuracy
        param_accuracy = compute_parameter_accuracy(predicted_params, ground_truth)
        
        # Save generated code
        code_path = CLIP_RESULTS_DIR / f'generated_code_pair_{pair_id:04d}.py'
        with open(code_path, 'w') as f:
            f.write(code)
        
        # Compile results
        result = {
            'pair_id': pair_id,
            'timestamp': datetime.now().isoformat(),
            'method': 'CLIP-Enhanced',
            'scores': scores,
            'predicted_physics': predicted_params,
            'ground_truth_physics': ground_truth.get('goal_physics', {}),
            'parameter_accuracy': param_accuracy,
            'code_path': str(code_path),
            'code_length': len(code),
            'code_lines': len(code.splitlines()),
            'code_valid': True,
        }
        
        # Add code quality metrics
        result['code_quality'] = {
            'has_imports': 'import bpy' in code,
            'has_physics': 'rigidbody' in code.lower(),
            'has_camera': 'camera' in code.lower(),
            'has_lighting': 'light' in code.lower() or 'sun' in code.lower(),
            'has_objects': ('sphere' in code.lower() or 'cube' in code.lower() or 'mesh' in code.lower()),
        }
        
        print(f"\n✅ Pair {pair_id} Results (CLIP-Enhanced):")
        print(f"   Per-Frame Fidelity: {scores.get('per_frame_fidelity', 0):.3f}")
        print(f"   TFA: {scores.get('temporal_flow_alignment', 0):.3f}")
        print(f"   PVS: {scores.get('physics_validity_score', 0):.3f}")
        print(f"   Code saved: {code_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pair {pair_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'pair_id': pair_id,
            'method': 'CLIP-Enhanced',
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
        }


def compute_statistics(results: List[Dict]) -> Dict:
    """Compute summary statistics across all results"""
    valid_results = [r for r in results if 'scores' in r and 'error' not in r]
    
    if not valid_results:
        return {}
    
    stats = {}
    
    # Score statistics
    for metric in ['per_frame_fidelity', 'temporal_flow_alignment', 'physics_validity_score']:
        values = [r['scores'].get(metric, 0) for r in valid_results]
        if values:
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }
    
    # Parameter accuracy statistics
    param_keys = ['ball_mass', 'ball_friction', 'track_friction', 'restitution', 'ramp_angle']
    for key in param_keys:
        accuracy_key = f'{key}_accuracy'
        values = [r['parameter_accuracy'].get(accuracy_key, 0) 
                 for r in valid_results if accuracy_key in r.get('parameter_accuracy', {})]
        if values:
            stats[accuracy_key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
    
    # Code quality statistics
    code_lengths = [r.get('code_length', 0) for r in valid_results]
    if code_lengths:
        stats['code_length'] = {
            'mean': float(np.mean(code_lengths)),
            'std': float(np.std(code_lengths)),
            'min': int(np.min(code_lengths)),
            'max': int(np.max(code_lengths)),
        }
    
    # Success rate
    stats['success_rate'] = len(valid_results) / len(results) if results else 0.0
    
    return stats


def main():
    """Main evaluation function with CLIP"""
    print("\n" + "="*70)
    print("CVPR Results Generation: CLIP-Enhanced Version")
    print("="*70)
    print(f"Results will be saved to: {CLIP_RESULTS_DIR}")
    print(f"Evaluating {NUM_TEST_PAIRS} test pairs")
    print(f"Convergence threshold: PFF > {CONVERGENCE_THRESHOLD} or TFA > 0.95")
    print(f"Max iterations: {MAX_ITERATIONS} (will stop early on convergence)")
    print("="*70 + "\n")
    
    # Initialize enhanced pipeline
    print("[Initialization] Setting up CLIP-enhanced pipeline...")
    coder = EnhancedZeroShotWorldCoder(
        vjepa_model_path=VJEPA_MODEL_PATH,
        llm_api_key=GEMINI_API_KEY,
        llm_provider='gemini',
        llm_model='gemini-2.0-flash-exp',
        max_iterations=MAX_ITERATIONS,
        use_clip=True,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        min_iterations=1,
        convergence_patience=2
    )
    print("  ✅ Pipeline initialized\n")
    
    # Evaluate all pairs
    all_results = []
    for pair_id in range(1, NUM_TEST_PAIRS + 1):
        result = evaluate_pair_with_clip(pair_id, coder)
        if result:
            all_results.append(result)
        
        # Save intermediate results
        intermediate_path = CLIP_RESULTS_DIR / f'intermediate_results_pair_{pair_id:04d}.json'
        with open(intermediate_path, 'w') as f:
            json.dump(convert_to_native_types(result), f, indent=2)
    
    # Compute statistics
    print("\n" + "="*70)
    print("Computing Statistics")
    print("="*70)
    stats = compute_statistics(all_results)
    
    # Save all results
    results_path = CLIP_RESULTS_DIR / 'all_results_clip.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_native_types({
            'timestamp': datetime.now().isoformat(),
            'method': 'CLIP-Enhanced',
            'num_pairs': NUM_TEST_PAIRS,
            'convergence_threshold': CONVERGENCE_THRESHOLD,
            'max_iterations': MAX_ITERATIONS,
            'results': all_results,
            'statistics': stats,
        }), f, indent=2)
    
    print(f"\n✅ All results saved to: {results_path}")
    
    # Generate summary report
    summary_path = CLIP_RESULTS_DIR / 'summary_report_clip.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CVPR Paper Results Summary: CLIP-Enhanced Version\n")
        f.write("="*70 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: CLIP-Enhanced (V-JEPA + CLIP + LLM)\n")
        f.write(f"Number of Test Pairs: {NUM_TEST_PAIRS}\n")
        f.write(f"Convergence Threshold: PFF > {CONVERGENCE_THRESHOLD} or TFA > 0.95\n")
        f.write(f"Max Iterations: {MAX_ITERATIONS} (convergence-based stopping)\n")
        f.write(f"Successful Evaluations: {len([r for r in all_results if 'scores' in r])}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-"*70 + "\n")
        for metric, values in stats.items():
            if isinstance(values, dict):
                f.write(f"\n{metric}:\n")
                for key, value in values.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"{metric}: {values}\n")
        
        f.write("\n\nPer-Pair Results:\n")
        f.write("-"*70 + "\n")
        for r in all_results:
            if 'scores' in r:
                f.write(f"\nPair {r['pair_id']}:\n")
                for key, value in r['scores'].items():
                    f.write(f"  {key}: {value:.3f}\n")
    
    print(f"✅ Summary report saved to: {summary_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY (CLIP-Enhanced)")
    print("="*70)
    if 'per_frame_fidelity' in stats:
        print(f"\nPer-Frame Fidelity: {stats['per_frame_fidelity']['mean']:.3f} ± {stats['per_frame_fidelity']['std']:.3f}")
        print(f"Temporal Flow Alignment: {stats['temporal_flow_alignment']['mean']:.3f} ± {stats['temporal_flow_alignment']['std']:.3f}")
        print(f"Physics Validity Score: {stats['physics_validity_score']['mean']:.3f} ± {stats['physics_validity_score']['std']:.3f}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        
        # Compare with baseline
        print(f"\n📊 Comparison with Baseline (Temporal-only):")
        baseline_pff = 0.245
        baseline_tfa = 1.000
        baseline_pvs = 1.000
        
        clip_pff = stats['per_frame_fidelity']['mean']
        clip_tfa = stats['temporal_flow_alignment']['mean']
        clip_pvs = stats['physics_validity_score']['mean']
        
        print(f"  PFF: {baseline_pff:.3f} → {clip_pff:.3f} (improvement: {clip_pff - baseline_pff:+.3f})")
        print(f"  TFA: {baseline_tfa:.3f} → {clip_tfa:.3f} (change: {clip_tfa - baseline_tfa:+.3f})")
        print(f"  PVS: {baseline_pvs:.3f} → {clip_pvs:.3f} (change: {clip_pvs - baseline_pvs:+.3f})")
    
    print(f"\n✅ All results saved to: {CLIP_RESULTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

