"""
Test Full Pipeline with Gemini API Key
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.zero_shot_worldcoder import ZeroShotWorldCoder

# Your Gemini API key
GEMINI_API_KEY = "AIzaSyBI52gm0JaJbMbM0xeqO9EuN86p88gIHj0"

def test_full_pipeline():
    """Test complete pipeline with V-JEPA + Gemini + Physics Verifier"""
    print("="*70)
    print("Full Pipeline Test: V-JEPA + Gemini + Physics Verifier")
    print("="*70)
    
    # Initialize with Gemini
    coder = ZeroShotWorldCoder(
        vjepa_model_path=str(PROJECT_ROOT / 'models' / 'vjepa' / 'vitl16.pth.tar'),
        llm_api_key=GEMINI_API_KEY,
        llm_provider='gemini',
        llm_model='gemini-2.0-flash-exp',
        max_iterations=2  # Start with 2 for testing
    )
    
    # Create test videos (or use your Blender scenes)
    print("\n[Step 0] Preparing test videos...")
    start_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 150 + 100
    goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 150 + 100
    
    print(f"   Start video shape: {start_video.shape}")
    print(f"   Goal video shape: {goal_video.shape}")
    
    # Run transformation
    print("\n[Step 1] Running full pipeline...")
    code, scores = coder.transform(start_video, goal_video)
    
    # Save generated code
    output_path = str(PROJECT_ROOT / 'examples' / 'generated_code_gemini.py')
    with open(output_path, 'w') as f:
        f.write(code)
    
    print(f"\n✅ Generated code saved to: {output_path}")
    print(f"\n📊 Final Scores:")
    for key, value in scores.items():
        print(f"   {key}: {value:.3f}")
    
    print(f"\n📝 Generated Code Preview (first 500 chars):")
    print("-"*70)
    print(code[:500])
    if len(code) > 500:
        print("...")
    print("-"*70)
    
    return code, scores


if __name__ == "__main__":
    code, scores = test_full_pipeline()
    
    print("\n" + "="*70)
    print("✅ FULL PIPELINE TEST COMPLETE!")
    print("="*70)
    print("\n💡 Next steps:")
    print("   1. Test with your actual Blender scene videos")
    print("   2. Check generated_code_gemini.py for the output")
    print("   3. Execute in Blender to verify")
    print("="*70 + "\n")





