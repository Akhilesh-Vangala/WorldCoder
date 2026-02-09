"""
Batch Evaluation Script for Ablation Studies
============================================
Runs evaluations with different configurations for ablation studies.
"""

import sys
sys.path.insert(0, '/Users/akhileshvangala/Desktop/CVPR')

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import subprocess

from generate_cvpr_results import evaluate_pair, render_blend_file, load_ground_truth_physics
from zero_shot_worldcoder import ZeroShotWorldCoder

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBI52gm0JaJbMbM0xeqO9EuN86p88gIHj0")
VJEPA_MODEL_PATH = '/Users/akhileshvangala/Desktop/CVPR/models/vjepa/vitl16.pth.tar'
DATASET_DIR = Path('/Users/akhileshvangala/Desktop/CVPR/dataset')
RESULTS_DIR = Path('/Users/akhileshvangala/Desktop/CVPR/cvpr_results')
ABLATION_DIR = RESULTS_DIR / 'ablation_studies'
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

NUM_TEST_PAIRS = 5


# Ablation configurations
ABLATION_CONFIGS = {
    'baseline': {
        'max_iterations': 3,
        'llm_provider': 'gemini',
        'llm_model': 'gemini-2.0-flash-exp',
        'use_vjepa': True,
        'use_physics_verifier': True,
    },
    'no_iterations': {
        'max_iterations': 1,
        'llm_provider': 'gemini',
        'llm_model': 'gemini-2.0-flash-exp',
        'use_vjepa': True,
        'use_physics_verifier': True,
    },
    'no_physics_verifier': {
        'max_iterations': 3,
        'llm_provider': 'gemini',
        'llm_model': 'gemini-2.0-flash-exp',
        'use_vjepa': True,
        'use_physics_verifier': False,  # Skip verification
    },
    'no_vjepa': {
        'max_iterations': 3,
        'llm_provider': 'gemini',
        'llm_model': 'gemini-2.0-flash-exp',
        'use_vjepa': False,  # Use placeholder embeddings
        'use_physics_verifier': True,
    },
}


def run_ablation_study(config_name: str, config: Dict):
    """Run evaluation with a specific configuration"""
    print("\n" + "="*70)
    print(f"Ablation Study: {config_name}")
    print("="*70)
    print(f"Configuration: {config}")
    print("="*70 + "\n")
    
    # Initialize pipeline with config
    if config['use_vjepa']:
        vjepa_path = VJEPA_MODEL_PATH
    else:
        vjepa_path = None  # Will use placeholder
    
    coder = ZeroShotWorldCoder(
        vjepa_model_path=vjepa_path,
        llm_api_key=GEMINI_API_KEY,
        llm_provider=config['llm_provider'],
        llm_model=config['llm_model'],
        max_iterations=config['max_iterations']
    )
    
    # Disable physics verifier if needed (modify the verifier to skip)
    if not config['use_physics_verifier']:
        # Create a dummy verifier that always passes
        class DummyVerifier:
            def verify_code(self, code, start_video, goal_video):
                return True, {
                    'per_frame_fidelity': 0.5,
                    'temporal_flow_alignment': 0.5,
                    'physics_validity_score': 0.5
                }, "Verification skipped"
        coder.verifier = DummyVerifier()
    
    # Evaluate all pairs
    all_results = []
    for pair_id in range(1, NUM_TEST_PAIRS + 1):
        print(f"\nEvaluating Pair {pair_id} with {config_name}...")
        
        # Load videos
        start_blend = DATASET_DIR / 'blender_files' / f'start_{pair_id:04d}.blend'
        goal_blend = DATASET_DIR / 'blender_files' / f'goal_{pair_id:04d}.blend'
        
        if not start_blend.exists() or not goal_blend.exists():
            print(f"⚠️  Files not found for pair {pair_id}, skipping...")
            continue
        
        start_video = render_blend_file(str(start_blend), num_frames=30)
        goal_video = render_blend_file(str(goal_blend), num_frames=30)
        
        # Run transformation
        try:
            code, scores = coder.transform(start_video, goal_video)
            
            result = {
                'pair_id': pair_id,
                'config': config_name,
                'scores': scores,
                'code_length': len(code),
                'timestamp': datetime.now().isoformat(),
            }
            all_results.append(result)
            
            print(f"  ✅ Pair {pair_id}: PFF={scores.get('per_frame_fidelity', 0):.3f}, "
                  f"TFA={scores.get('temporal_flow_alignment', 0):.3f}, "
                  f"PVS={scores.get('physics_validity_score', 0):.3f}")
            
        except Exception as e:
            print(f"  ❌ Pair {pair_id} failed: {e}")
            all_results.append({
                'pair_id': pair_id,
                'config': config_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            })
    
    # Save results
    results_path = ABLATION_DIR / f'{config_name}_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config_name': config_name,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    # Compute statistics
    valid_results = [r for r in all_results if 'scores' in r]
    if valid_results:
        import numpy as np
        stats = {}
        for metric in ['per_frame_fidelity', 'temporal_flow_alignment', 'physics_validity_score']:
            values = [r['scores'].get(metric, 0) for r in valid_results]
            if values:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        
        print(f"\nStatistics for {config_name}:")
        for metric, values in stats.items():
            print(f"  {metric}: {values['mean']:.3f} ± {values['std']:.3f}")
        
        return stats
    
    return None


def compare_ablation_results():
    """Compare results across all ablation configurations"""
    print("\n" + "="*70)
    print("Ablation Study Comparison")
    print("="*70 + "\n")
    
    all_stats = {}
    
    # Load all results
    for config_name in ABLATION_CONFIGS.keys():
        results_path = ABLATION_DIR / f'{config_name}_results.json'
        if results_path.exists():
            with open(results_path, 'r') as f:
                data = json.load(f)
                results = data['results']
                
                # Compute stats
                valid_results = [r for r in results if 'scores' in r]
                if valid_results:
                    import numpy as np
                    stats = {}
                    for metric in ['per_frame_fidelity', 'temporal_flow_alignment', 'physics_validity_score']:
                        values = [r['scores'].get(metric, 0) for r in valid_results]
                        if values:
                            stats[metric] = {
                                'mean': float(np.mean(values)),
                                'std': float(np.std(values)),
                            }
                    all_stats[config_name] = stats
    
    # Create comparison table
    print("Comparison Table:")
    print("-"*70)
    print(f"{'Config':<25} {'PFF':<12} {'TFA':<12} {'PVS':<12}")
    print("-"*70)
    
    for config_name, stats in all_stats.items():
        pff = stats.get('per_frame_fidelity', {}).get('mean', 0)
        tfa = stats.get('temporal_flow_alignment', {}).get('mean', 0)
        pvs = stats.get('physics_validity_score', {}).get('mean', 0)
        print(f"{config_name:<25} {pff:<12.3f} {tfa:<12.3f} {pvs:<12.3f}")
    
    # Save comparison
    comparison_path = ABLATION_DIR / 'ablation_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n✅ Comparison saved to: {comparison_path}")
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[t]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Ablation Study Results}\n"
    latex_table += "\\label{tab:ablation}\n"
    latex_table += "\\begin{tabular}{lccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Configuration & PFF & TFA & PVS \\\\\n"
    latex_table += "\\midrule\n"
    
    for config_name, stats in all_stats.items():
        pff = stats.get('per_frame_fidelity', {}).get('mean', 0)
        tfa = stats.get('temporal_flow_alignment', {}).get('mean', 0)
        pvs = stats.get('physics_validity_score', {}).get('mean', 0)
        latex_table += f"{config_name.replace('_', ' ').title()} & {pff:.3f} & {tfa:.3f} & {pvs:.3f} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    latex_path = ABLATION_DIR / 'ablation_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✅ LaTeX table saved to: {latex_path}\n")


def main():
    """Run all ablation studies"""
    print("\n" + "="*70)
    print("Batch Evaluation: Ablation Studies")
    print("="*70)
    print(f"Results will be saved to: {ABLATION_DIR}")
    print(f"Number of configurations: {len(ABLATION_CONFIGS)}")
    print("="*70 + "\n")
    
    # Run each ablation study
    for config_name, config in ABLATION_CONFIGS.items():
        try:
            run_ablation_study(config_name, config)
        except Exception as e:
            print(f"❌ Ablation study {config_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare results
    compare_ablation_results()
    
    print("\n" + "="*70)
    print("✅ All ablation studies complete!")
    print(f"Results saved to: {ABLATION_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

