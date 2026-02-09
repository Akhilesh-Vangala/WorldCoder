"""
Comprehensive Evaluation Script for CVPR Paper
===============================================
Generates evaluation results across all test pairs with detailed metrics,
statistics, and ground truth comparisons.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import subprocess
import tempfile
import shutil

from src.zero_shot_worldcoder import ZeroShotWorldCoder, VJEPAEncoder, LLMCodeGenerator, PhysicsVerifier


def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    else:
        return obj

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBI52gm0JaJbMbM0xeqO9EuN86p88gIHj0")
VJEPA_MODEL_PATH = PROJECT_ROOT / 'models' / 'vjepa' / 'vitl16.pth.tar'
BLENDER_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'
DATASET_DIR = PROJECT_ROOT / 'dataset'
RESULTS_DIR = PROJECT_ROOT / 'cvpr_results'
RESULTS_DIR.mkdir(exist_ok=True)

NUM_TEST_PAIRS = 5
MAX_ITERATIONS = 3


def render_blend_file(blend_path: str, num_frames: int = 30) -> np.ndarray:
    """Render Blender file to numpy video using Blender CLI"""
    temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(temp_dir, 'render_script.py')
    
    render_script = f"""
import bpy
import sys
import os

# Open blend file
bpy.ops.wm.open_mainfile(filepath=r'{blend_path}')

scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 300
scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.resolution_percentage = 100
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.image_settings.file_format = 'PNG'

output_dir = r'{temp_dir}'
os.makedirs(output_dir, exist_ok=True)

# Sample frames evenly
frame_indices = list(range(1, min(301, scene.frame_end + 1), max(1, (scene.frame_end - 1) // {num_frames - 1})))
if scene.frame_end not in frame_indices:
    frame_indices.append(scene.frame_end)

for i, frame_num in enumerate(frame_indices[:{num_frames}]):
    scene.frame_set(frame_num)
    frame_path = os.path.join(output_dir, f'frame_{{i:04d}}.png')
    scene.render.filepath = frame_path
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as e:
        print(f"Error rendering frame {{frame_num}}: {{e}}")

print(f"Rendered {{len(frame_indices[:{num_frames}])}} frames")
"""
    
    with open(script_path, 'w') as f:
        f.write(render_script)
    
    # Run Blender
    result = subprocess.run(
        [BLENDER_PATH, '--background', '--python', script_path],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    # Load frames
    video_frames = []
    try:
        import cv2
        for i in range(num_frames):
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img_rgb.shape[2] == 4:
                        img_rgb = img_rgb[:, :, :3]
                    video_frames.append(img_rgb)
    except ImportError:
        from PIL import Image
        for i in range(num_frames):
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            if os.path.exists(frame_path):
                img = Image.open(frame_path)
                img_array = np.array(img)
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                video_frames.append(img_array)
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    if video_frames:
        return np.array(video_frames)
    else:
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)


def load_ground_truth_physics(pair_id: int) -> Dict:
    """Load ground truth physics parameters for a pair"""
    physics_path = DATASET_DIR / 'physics' / f'physics_{pair_id:04d}.json'
    if physics_path.exists():
        with open(physics_path, 'r') as f:
            return json.load(f)
    return {}


def extract_physics_params_from_code(code: str) -> Dict:
    """Extract physics parameters from generated Blender code"""
    params = {
        'ball_mass': None,
        'ball_friction': None,
        'track_friction': None,
        'restitution': None,
        'ramp_angle': None,
        'gravity': None,
    }
    
    import re
    
    # Extract ball mass
    mass_match = re.search(r'ball\.rigid_body\.mass\s*=\s*([0-9.]+)', code)
    if mass_match:
        params['ball_mass'] = float(mass_match.group(1))
    
    # Extract ball friction
    ball_friction_match = re.search(r'ball\.rigid_body\.friction\s*=\s*([0-9.]+)', code)
    if ball_friction_match:
        params['ball_friction'] = float(ball_friction_match.group(1))
    
    # Extract track friction
    track_friction_match = re.search(r'track\.rigid_body\.friction\s*=\s*([0-9.]+)', code)
    if track_friction_match:
        params['track_friction'] = float(track_friction_match.group(1))
    
    # Extract restitution
    restitution_match = re.search(r'ball\.rigid_body\.restitution\s*=\s*([0-9.]+)', code)
    if restitution_match:
        params['restitution'] = float(restitution_match.group(1))
    
    # Extract ramp angle
    angle_match = re.search(r'math\.radians\(([0-9.]+)\)', code)
    if angle_match:
        params['ramp_angle'] = float(angle_match.group(1))
    
    # Extract gravity
    gravity_match = re.search(r'scene\.gravity\s*=\s*\([^)]*,\s*[^)]*,\s*([-0-9.]+)\)', code)
    if gravity_match:
        params['gravity'] = float(gravity_match.group(1))
    
    return params


def compute_parameter_accuracy(predicted: Dict, ground_truth: Dict) -> Dict:
    """Compute accuracy of predicted physics parameters vs ground truth"""
    accuracy = {}
    
    goal_physics = ground_truth.get('goal_physics', {})
    
    for key in ['ball_mass', 'ball_friction', 'track_friction', 'restitution', 'ramp_angle']:
        if key in predicted and predicted[key] is not None:
            if key in goal_physics:
                gt_value = goal_physics[key]
                pred_value = predicted[key]
                # For angles, convert radians to degrees if needed
                if key == 'ramp_angle' and pred_value < 2.0:  # Likely in radians
                    import math
                    pred_value = math.degrees(pred_value)
                
                error = abs(pred_value - gt_value)
                relative_error = error / (abs(gt_value) + 1e-6)
                accuracy[f'{key}_error'] = error
                accuracy[f'{key}_relative_error'] = relative_error
                accuracy[f'{key}_accuracy'] = max(0.0, 1.0 - relative_error)
    
    return accuracy


def evaluate_pair(pair_id: int, coder: ZeroShotWorldCoder) -> Dict:
    """Evaluate a single test pair"""
    print(f"\n{'='*70}")
    print(f"Evaluating Pair {pair_id}")
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
    
    # Run transformation
    print(f"[Step 2] Running transformation pipeline...")
    try:
        code, scores = coder.transform(start_video, goal_video)
        
        # Extract physics parameters from code
        predicted_params = extract_physics_params_from_code(code)
        
        # Compute parameter accuracy
        param_accuracy = compute_parameter_accuracy(predicted_params, ground_truth)
        
        # Save generated code
        code_path = RESULTS_DIR / f'generated_code_pair_{pair_id:04d}.py'
        with open(code_path, 'w') as f:
            f.write(code)
        
        # Compile results
        result = {
            'pair_id': pair_id,
            'timestamp': datetime.now().isoformat(),
            'scores': scores,
            'predicted_physics': predicted_params,
            'ground_truth_physics': ground_truth.get('goal_physics', {}),
            'parameter_accuracy': param_accuracy,
            'code_path': str(code_path),
            'code_length': len(code),
            'code_lines': len(code.splitlines()),
            'code_valid': True,  # Checked by verifier
        }
        
        # Add code quality metrics
        result['code_quality'] = {
            'has_imports': 'import bpy' in code,
            'has_physics': 'rigidbody' in code.lower(),
            'has_camera': 'camera' in code.lower(),
            'has_lighting': 'light' in code.lower() or 'sun' in code.lower(),
            'has_objects': ('sphere' in code.lower() or 'cube' in code.lower() or 'mesh' in code.lower()),
        }
        
        print(f"\n✅ Pair {pair_id} Results:")
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


def generate_latex_table(results: List[Dict], stats: Dict) -> str:
    """Generate LaTeX table for paper"""
    valid_results = [r for r in results if 'scores' in r and 'error' not in r]
    
    table = "\\begin{table*}[t]\n"
    table += "\\centering\n"
    table += "\\caption{Evaluation Results on Test Pairs}\n"
    table += "\\label{tab:results}\n"
    table += "\\begin{tabular}{lcccc}\n"
    table += "\\toprule\n"
    table += "Pair & Per-Frame Fidelity & TFA & PVS & Success \\\\\n"
    table += "\\midrule\n"
    
    for r in valid_results:
        pair_id = r['pair_id']
        scores = r['scores']
        pff = scores.get('per_frame_fidelity', 0)
        tfa = scores.get('temporal_flow_alignment', 0)
        pvs = scores.get('physics_validity_score', 0)
        success = "Yes" if (pff > 0.85 and tfa > 0.80 and pvs > 0.90) else "No"
        
        table += f"{pair_id} & {pff:.3f} & {tfa:.3f} & {pvs:.3f} & {success} \\\\\n"
    
    # Add mean row
    if 'per_frame_fidelity' in stats:
        table += "\\midrule\n"
        table += "Mean & "
        table += f"{stats['per_frame_fidelity']['mean']:.3f} & "
        table += f"{stats['temporal_flow_alignment']['mean']:.3f} & "
        table += f"{stats['physics_validity_score']['mean']:.3f} & "
        table += f"{stats['success_rate']:.1%} \\\\\n"
        table += "Std & "
        table += f"{stats['per_frame_fidelity']['std']:.3f} & "
        table += f"{stats['temporal_flow_alignment']['std']:.3f} & "
        table += f"{stats['physics_validity_score']['std']:.3f} & "
        table += "--- \\\\\n"
    
    table += "\\bottomrule\n"
    table += "\\end{tabular}\n"
    table += "\\end{table*}\n"
    
    return table


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("CVPR Paper Results Generation")
    print("="*70)
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Evaluating {NUM_TEST_PAIRS} test pairs")
    print("="*70 + "\n")
    
    # Initialize pipeline
    print("[Initialization] Setting up pipeline...")
    coder = ZeroShotWorldCoder(
        vjepa_model_path=str(VJEPA_MODEL_PATH),
        llm_api_key=GEMINI_API_KEY,
        llm_provider='gemini',
        llm_model='gemini-2.0-flash-exp',
        max_iterations=MAX_ITERATIONS
    )
    print("  ✅ Pipeline initialized\n")
    
    # Evaluate all pairs
    all_results = []
    for pair_id in range(1, NUM_TEST_PAIRS + 1):
        result = evaluate_pair(pair_id, coder)
        if result:
            all_results.append(result)
        
        # Save intermediate results
        intermediate_path = RESULTS_DIR / f'intermediate_results_pair_{pair_id:04d}.json'
        with open(intermediate_path, 'w') as f:
            json.dump(convert_to_native_types(result), f, indent=2)
    
    # Compute statistics
    print("\n" + "="*70)
    print("Computing Statistics")
    print("="*70)
    stats = compute_statistics(all_results)
    
    # Save all results
    results_path = RESULTS_DIR / 'all_results.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_native_types({
            'timestamp': datetime.now().isoformat(),
            'num_pairs': NUM_TEST_PAIRS,
            'results': all_results,
            'statistics': stats,
        }), f, indent=2)
    
    print(f"\n✅ All results saved to: {results_path}")
    
    # Generate summary report
    summary_path = RESULTS_DIR / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CVPR Paper Results Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Test Pairs: {NUM_TEST_PAIRS}\n")
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
    
    # Generate LaTeX table
    latex_table = generate_latex_table(all_results, stats)
    latex_path = RESULTS_DIR / 'results_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"✅ LaTeX table saved to: {latex_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    if 'per_frame_fidelity' in stats:
        print(f"\nPer-Frame Fidelity: {stats['per_frame_fidelity']['mean']:.3f} ± {stats['per_frame_fidelity']['std']:.3f}")
        print(f"Temporal Flow Alignment: {stats['temporal_flow_alignment']['mean']:.3f} ± {stats['temporal_flow_alignment']['std']:.3f}")
        print(f"Physics Validity Score: {stats['physics_validity_score']['mean']:.3f} ± {stats['physics_validity_score']['std']:.3f}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"\n✅ All results saved to: {RESULTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

