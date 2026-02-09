"""
Quick test to check if the code is ready for use
Tests without requiring video rendering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.zero_shot_worldcoder import ZeroShotWorldCoder, VJEPAEncoder, LLMCodeGenerator, PhysicsVerifier

def test_readiness():
    """Test if all components work and code quality"""
    print("="*70)
    print("Zero-Shot WorldCoder - Readiness Check")
    print("="*70)
    
    issues = []
    warnings = []
    
    # Test 1: Component Initialization
    print("\n[Test 1] Component Initialization")
    try:
        vjepa = VJEPAEncoder()
        llm = LLMCodeGenerator()
        verifier = PhysicsVerifier()
        coder = ZeroShotWorldCoder()
        print("✅ All components initialize successfully")
    except Exception as e:
        issues.append(f"Initialization failed: {e}")
        print(f"❌ Initialization failed: {e}")
    
    # Test 2: Embedding Generation
    print("\n[Test 2] Embedding Generation")
    try:
        test_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
        start_emb = vjepa.encode(test_video)
        goal_emb = vjepa.encode(test_video)
        print(f"✅ Embeddings generated: {start_emb.shape}")
        if start_emb.shape[0] < 100:
            warnings.append("Embedding dimension seems low (< 100)")
    except Exception as e:
        issues.append(f"Embedding generation failed: {e}")
        print(f"❌ Failed: {e}")
    
    # Test 3: Code Generation
    print("\n[Test 3] Code Generation")
    try:
        code = llm.generate_code(start_emb, goal_emb)
        print(f"✅ Code generated: {len(code)} chars")
        
        # Quality checks
        has_bpy = 'bpy' in code
        has_physics = 'rigidbody' in code.lower() or 'physics' in code.lower()
        has_structure = 'import' in code or 'def' in code or '=' in code
        
        print(f"   - Has bpy: {has_bpy}")
        print(f"   - Has physics: {has_physics}")
        print(f"   - Has structure: {has_structure}")
        
        if not has_bpy:
            issues.append("Generated code missing 'bpy' import")
        if not has_physics:
            warnings.append("Generated code might be missing physics setup")
        if not has_structure:
            issues.append("Generated code seems empty/invalid")
            
    except Exception as e:
        issues.append(f"Code generation failed: {e}")
        print(f"❌ Failed: {e}")
    
    # Test 4: Full Pipeline (without video rendering)
    print("\n[Test 4] Full Pipeline (Synthetic)")
    try:
        start_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
        goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
        
        code, scores = coder.transform(start_video, goal_video)
        print("✅ Pipeline executed successfully")
        print(f"   Code length: {len(code)} chars")
        print(f"   Scores: {scores}")
        
        # Check if scores are reasonable
        if scores:
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            if avg_score < 0.5:
                warnings.append("Scores seem low (might be placeholder values)")
    except Exception as e:
        issues.append(f"Pipeline execution failed: {e}")
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Code Syntax
    print("\n[Test 5] Code Syntax Validation")
    try:
        if 'code' in locals() and code:
            compile(code, '<string>', 'exec')
            print("✅ Generated code is syntactically valid")
        else:
            warnings.append("No code to validate")
    except SyntaxError as e:
        issues.append(f"Generated code has syntax errors: {e}")
        print(f"❌ Syntax error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("READINESS SUMMARY")
    print("="*70)
    
    if not issues:
        print("\n✅ READY TO USE!")
        print("   All core components work. You can:")
        print("   1. Test with real videos")
        print("   2. Add OpenAI API key for better LLM generation")
        print("   3. Load real V-JEPA model (optional)")
    else:
        print(f"\n❌ {len(issues)} CRITICAL ISSUE(S):")
        for issue in issues:
            print(f"   - {issue}")
        print("\n⚠️  Fix these before using in production")
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S):")
        for warning in warnings:
            print(f"   - {warning}")
        print("\n💡 These are suggestions but won't block usage")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("\nTo improve code quality:")
    print("1. ✅ Set OPENAI_API_KEY for real LLM generation")
    print("2. ✅ Load V-JEPA model for better embeddings")
    print("3. ✅ Test with actual Blender scene videos")
    print("4. ✅ Implement proper video loading in _load_video()")
    print("5. ✅ Enhance PhysicsVerifier._execute_and_render()")
    
    return len(issues) == 0


if __name__ == "__main__":
    ready = test_readiness()
    exit(0 if ready else 1)





