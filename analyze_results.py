"""
Results Analysis Script for CVPR Paper
=======================================
Analyzes saved results and generates comparison plots, tables, and insights.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available. Plots will be skipped.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available. CSV export will be skipped.")

RESULTS_DIR = Path('/Users/akhileshvangala/Desktop/CVPR/cvpr_results')


def load_results() -> Dict:
    """Load all results from JSON file"""
    results_path = RESULTS_DIR / 'all_results.json'
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def create_metrics_plot(results: Dict):
    """Create visualization of metrics across pairs"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping plot")
        return
    
    valid_results = [r for r in results['results'] if 'scores' in r and 'error' not in r]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    pairs = [r['pair_id'] for r in valid_results]
    pff = [r['scores'].get('per_frame_fidelity', 0) for r in valid_results]
    tfa = [r['scores'].get('temporal_flow_alignment', 0) for r in valid_results]
    pvs = [r['scores'].get('physics_validity_score', 0) for r in valid_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pairs))
    width = 0.25
    
    ax.bar(x - width, pff, width, label='Per-Frame Fidelity', alpha=0.8)
    ax.bar(x, tfa, width, label='Temporal Flow Alignment', alpha=0.8)
    ax.bar(x + width, pvs, width, label='Physics Validity Score', alpha=0.8)
    
    ax.set_xlabel('Test Pair')
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics Across Test Pairs')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Pair {p}' for p in pairs])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / 'metrics_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Metrics plot saved to: {plot_path}")
    plt.close()


def create_parameter_comparison(results: Dict):
    """Create comparison of predicted vs ground truth physics parameters"""
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib not available, skipping plot")
        return
    
    valid_results = [r for r in results['results'] if 'predicted_physics' in r and 'ground_truth_physics' in r]
    
    if not valid_results:
        print("No parameter data to compare")
        return
    
    # Extract parameters
    params_to_compare = ['ball_mass', 'ball_friction', 'track_friction', 'restitution']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, param in enumerate(params_to_compare):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        predicted = []
        ground_truth = []
        pairs = []
        
        for r in valid_results:
            pred_val = r['predicted_physics'].get(param)
            gt_val = r['ground_truth_physics'].get(param)
            
            if pred_val is not None and gt_val is not None:
                predicted.append(pred_val)
                ground_truth.append(gt_val)
                pairs.append(r['pair_id'])
        
        if predicted:
            ax.scatter(ground_truth, predicted, s=100, alpha=0.6)
            # Add diagonal line (perfect prediction)
            min_val = min(min(ground_truth), min(predicted))
            max_val = max(max(ground_truth), max(predicted))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{param.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = RESULTS_DIR / 'parameter_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Parameter comparison plot saved to: {plot_path}")
    plt.close()


def create_summary_table(results: Dict):
    """Create summary table"""
    valid_results = [r for r in results['results'] if 'scores' in r and 'error' not in r]
    
    data = []
    for r in valid_results:
        row = {
            'Pair ID': r['pair_id'],
            'Per-Frame Fidelity': r['scores'].get('per_frame_fidelity', 0),
            'TFA': r['scores'].get('temporal_flow_alignment', 0),
            'PVS': r['scores'].get('physics_validity_score', 0),
            'Code Length': r.get('code_length', 0),
        }
        
        # Add parameter accuracies if available
        if 'parameter_accuracy' in r:
            for key in ['ball_mass_accuracy', 'ball_friction_accuracy', 'track_friction_accuracy']:
                if key in r['parameter_accuracy']:
                    row[key] = r['parameter_accuracy'][key]
        
        data.append(row)
    
    if PANDAS_AVAILABLE:
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = RESULTS_DIR / 'results_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"✅ Results table saved to: {csv_path}")
        
        # Print summary
        print("\nSummary Statistics:")
        print(df.describe())
        
        return df
    else:
        # Fallback: save as JSON
        csv_path = RESULTS_DIR / 'results_table.json'
        with open(csv_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Results table saved to: {csv_path}")
        
        # Print summary manually
        print("\nSummary Statistics:")
        if data:
            import numpy as np
            for key in ['Per-Frame Fidelity', 'TFA', 'PVS']:
                values = [r[key] for r in data if key in r]
                if values:
                    print(f"{key}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
        
        return data


def generate_insights(results: Dict):
    """Generate insights and analysis from results"""
    valid_results = [r for r in results['results'] if 'scores' in r and 'error' not in r]
    stats = results.get('statistics', {})
    
    insights = []
    
    # Overall performance
    if 'per_frame_fidelity' in stats:
        pff_mean = stats['per_frame_fidelity']['mean']
        insights.append(f"Average Per-Frame Fidelity: {pff_mean:.3f}")
        if pff_mean > 0.85:
            insights.append("✅ Excellent visual fidelity achieved")
        elif pff_mean > 0.70:
            insights.append("⚠️  Good visual fidelity, room for improvement")
        else:
            insights.append("❌ Visual fidelity needs improvement")
    
    if 'temporal_flow_alignment' in stats:
        tfa_mean = stats['temporal_flow_alignment']['mean']
        insights.append(f"Average Temporal Flow Alignment: {tfa_mean:.3f}")
        if tfa_mean > 0.80:
            insights.append("✅ Excellent temporal consistency")
        elif tfa_mean > 0.65:
            insights.append("⚠️  Good temporal consistency")
        else:
            insights.append("❌ Temporal consistency needs improvement")
    
    if 'physics_validity_score' in stats:
        pvs_mean = stats['physics_validity_score']['mean']
        insights.append(f"Average Physics Validity Score: {pvs_mean:.3f}")
        if pvs_mean > 0.90:
            insights.append("✅ Excellent physics validity")
        elif pvs_mean > 0.75:
            insights.append("⚠️  Good physics validity")
        else:
            insights.append("❌ Physics validity needs improvement")
    
    # Success rate
    success_rate = stats.get('success_rate', 0)
    insights.append(f"\nOverall Success Rate: {success_rate:.1%}")
    if success_rate >= 0.8:
        insights.append("✅ High success rate across test pairs")
    elif success_rate >= 0.6:
        insights.append("⚠️  Moderate success rate")
    else:
        insights.append("❌ Success rate needs improvement")
    
    # Parameter accuracy
    param_accuracies = {}
    for r in valid_results:
        if 'parameter_accuracy' in r:
            for key, value in r['parameter_accuracy'].items():
                if key.endswith('_accuracy'):
                    param_name = key.replace('_accuracy', '')
                    if param_name not in param_accuracies:
                        param_accuracies[param_name] = []
                    param_accuracies[param_name].append(value)
    
    if param_accuracies:
        insights.append("\nParameter Prediction Accuracy:")
        for param, accs in param_accuracies.items():
            mean_acc = np.mean(accs)
            insights.append(f"  {param.replace('_', ' ').title()}: {mean_acc:.3f}")
    
    # Save insights
    insights_path = RESULTS_DIR / 'insights.txt'
    with open(insights_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Results Analysis Insights\n")
        f.write("="*70 + "\n\n")
        for insight in insights:
            f.write(insight + "\n")
    
    print(f"\n✅ Insights saved to: {insights_path}")
    
    # Print insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    for insight in insights:
        print(insight)
    print("="*70 + "\n")


def main():
    """Main analysis function"""
    print("\n" + "="*70)
    print("Results Analysis")
    print("="*70 + "\n")
    
    # Load results
    results = load_results()
    if not results:
        return
    
    print(f"Loaded results for {len(results['results'])} pairs\n")
    
    # Create visualizations
    print("[1] Creating metrics plot...")
    try:
        create_metrics_plot(results)
    except Exception as e:
        print(f"⚠️  Failed to create metrics plot: {e}")
    
    print("\n[2] Creating parameter comparison...")
    try:
        create_parameter_comparison(results)
    except Exception as e:
        print(f"⚠️  Failed to create parameter comparison: {e}")
    
    print("\n[3] Creating summary table...")
    try:
        df = create_summary_table(results)
    except Exception as e:
        print(f"⚠️  Failed to create summary table: {e}")
    
    print("\n[4] Generating insights...")
    try:
        generate_insights(results)
    except Exception as e:
        print(f"⚠️  Failed to generate insights: {e}")
    
    print("\n✅ Analysis complete!")
    print(f"All outputs saved to: {RESULTS_DIR}\n")


if __name__ == "__main__":
    main()

