"""
Test Enhanced WorldCoder with CLIP visual features
"""

import sys
sys.path.insert(0, '/Users/akhileshvangala/Desktop/CVPR')

import numpy as np
from zero_shot_worldcoder_enhanced import EnhancedZeroShotWorldCoder
import os

# Test with one pair
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBI52gm0JaJbMbM0xeqO9EuN86p88gIHj0")
VJEPA_MODEL_PATH = '/Users/akhileshvangala/Desktop/CVPR/models/vjepa/vitl16.pth.tar'

def test_enhanced_pipeline():
    """Test enhanced pipeline with CLIP"""
    print("="*70)
    print("Testing Enhanced WorldCoder with CLIP")
    print("="*70)
    
    # Initialize enhanced coder
    coder = EnhancedZeroShotWorldCoder(
        vjepa_model_path=VJEPA_MODEL_PATH,
        llm_api_key=GEMINI_API_KEY,
        llm_provider='gemini',
        llm_model='gemini-2.0-flash-exp',
        max_iterations=3,
        use_clip=True
    )
    
    # Load test videos (use existing blend files)
    from generate_cvpr_results import render_blend_file
    dataset_dir = '/Users/akhileshvangala/Desktop/CVPR/dataset/blender_files'
    
    start_blend = f"{dataset_dir}/start_0001.blend"
    goal_blend = f"{dataset_dir}/goal_0001.blend"
    
    print("\n[Loading videos]...")
    start_video = render_blend_file(start_blend, num_frames=30)
    goal_video = render_blend_file(goal_blend, num_frames=30)
    
    print(f"Start video: {start_video.shape}")
    print(f"Goal video: {goal_video.shape}")
    
    # Run transformation
    print("\n[Running transformation]...")
    code, scores = coder.transform(start_video, goal_video)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Per-Frame Fidelity: {scores.get('per_frame_fidelity', 0):.3f}")
    print(f"TFA: {scores.get('temporal_flow_alignment', 0):.3f}")
    print(f"PVS: {scores.get('physics_validity_score', 0):.3f}")
    print(f"\nCode length: {len(code)} characters")
    print("="*70)
    
    return code, scores

if __name__ == "__main__":
    test_enhanced_pipeline()

