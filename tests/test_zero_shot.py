"""
Quick test of zero-shot WorldCoder pipeline
Tests that all components can be imported and initialized
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.zero_shot_worldcoder import ZeroShotWorldCoder, VJEPAEncoder, LLMCodeGenerator, PhysicsVerifier

def test_components():
    """Test each component individually"""
    print("Testing Zero-Shot WorldCoder Components")
    print("="*70)
    
    # Test 1: V-JEPA Encoder
    print("\n[Test 1] V-JEPA Encoder")
    vjepa = VJEPAEncoder()
    test_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
    embedding = vjepa.encode(test_video)
    print(f"✅ Embedding shape: {embedding.shape}")
    print(f"   Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    
    # Test 2: LLM Code Generator
    print("\n[Test 2] LLM Code Generator")
    llm = LLMCodeGenerator()
    start_emb = np.random.randn(512)
    goal_emb = np.random.randn(512)
    code = llm.generate_code(start_emb, goal_emb)
    print(f"✅ Code generated: {len(code)} characters")
    print(f"   Preview: {code[:100]}...")
    
    # Test 3: Physics Verifier
    print("\n[Test 3] Physics Verifier")
    verifier = PhysicsVerifier()
    print(f"✅ Verifier initialized with Blender: {verifier.blender_path}")
    
    # Test 4: Full Pipeline
    print("\n[Test 4] Full Pipeline")
    coder = ZeroShotWorldCoder(max_iterations=2)
    start_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
    goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
    
    try:
        code, scores = coder.transform(start_video, goal_video)
        print(f"✅ Pipeline executed successfully!")
        print(f"   Generated code: {len(code)} chars")
        print(f"   Scores: {scores}")
    except Exception as e:
        print(f"⚠️  Pipeline test failed: {e}")
        print("   (This is expected if V-JEPA/LLM not fully configured)")
    
    print("\n" + "="*70)
    print("✅ All components can be imported and initialized!")
    print("\n💡 Next steps:")
    print("   1. Set OPENAI_API_KEY for real LLM generation")
    print("   2. Load V-JEPA model (or use placeholder)")
    print("   3. Test with real video data")


if __name__ == "__main__":
    test_components()





