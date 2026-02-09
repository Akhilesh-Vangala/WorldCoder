"""
Comprehensive Test of Zero-Shot WorldCoder Pipeline
Tests all components with real Blender scenes and detailed diagnostics
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.zero_shot_worldcoder import ZeroShotWorldCoder, VJEPAEncoder, LLMCodeGenerator, PhysicsVerifier
import subprocess
import tempfile
import os

GEMINI_API_KEY = "AIzaSyBI52gm0JaJbMbM0xeqO9EuN86p88gIHj0"

def render_blend_file(blend_path: str, num_frames: int = 30) -> np.ndarray:
    """Render Blender file to numpy video using Blender CLI"""
    temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(temp_dir, 'render_script.py')
    
    # Create rendering script
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

# Sample frames
frame_indices = list(range(1, min(301, scene.frame_end + 1), max(1, (scene.frame_end - 1) // {num_frames - 1})))
if scene.frame_end not in frame_indices:
    frame_indices.append(scene.frame_end)

for i, frame_num in enumerate(frame_indices[:{num_frames}]):
    scene.frame_set(frame_num)
    frame_path = os.path.join(output_dir, f'frame_{{i:04d}}.png')
    scene.render.filepath = frame_path
    try:
        bpy.ops.render.render(write_still=True)
        print(f"Rendered frame {{frame_num}} -> {{frame_path}}")
    except Exception as e:
        print(f"Error rendering frame {{frame_num}}: {{e}}")

print(f"Done. Rendered {{len(frame_indices[:{num_frames}])}} frames")
"""
    
    with open(script_path, 'w') as f:
        f.write(render_script)
    
    # Run Blender
    blender_path = '/Applications/Blender.app/Contents/MacOS/Blender'
    result = subprocess.run(
        [blender_path, '--background', '--python', script_path],
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
                    # Ensure RGB (3 channels), remove alpha if present
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
                # Convert RGBA to RGB if needed
                if img_array.shape[2] == 4:
                    img_array = img_array[:, :, :3]
                video_frames.append(img_array)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    if video_frames:
        video = np.array(video_frames)
        print(f"    ✅ Loaded {len(video_frames)} frames, shape: {video.shape}, brightness: {video.mean():.1f}")
        return video
    else:
        print(f"    ⚠️  No frames loaded, using placeholder")
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)


def test_individual_components():
    """Test each component separately"""
    print("="*70)
    print("COMPONENT TESTS")
    print("="*70)
    
    # Test 1: V-JEPA
    print("\n[Test 1] V-JEPA Encoder...")
    try:
        vjepa = VJEPAEncoder(str(PROJECT_ROOT / 'models' / 'vjepa' / 'vitl16.pth.tar'))
        test_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
        embedding = vjepa.encode(test_video)
        print(f"  ✅ Embedding shape: {embedding.shape}")
        print(f"  ✅ Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
    
    # Test 2: LLM Code Generator
    print("\n[Test 2] LLM Code Generator (Gemini)...")
    try:
        llm = LLMCodeGenerator(GEMINI_API_KEY, provider='gemini')
        test_emb = np.random.randn(1024)
        code = llm.generate_code(test_emb, test_emb)
        print(f"  ✅ Generated {len(code)} characters")
        print(f"  ✅ Code preview: {code[:150]}...")
        # Check code quality
        has_imports = 'import bpy' in code
        has_physics = 'rigidbody' in code.lower() or 'gravity' in code.lower()
        print(f"  ✅ Has bpy import: {has_imports}")
        print(f"  ✅ Has physics: {has_physics}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Physics Verifier
    print("\n[Test 3] Physics Verifier...")
    try:
        verifier = PhysicsVerifier()
        # Simple test code
        test_code = """
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
"""
        test_start = np.random.rand(30, 224, 224, 3).astype(np.uint8)
        test_goal = np.random.rand(30, 224, 224, 3).astype(np.uint8)
        
        is_valid, scores, feedback = verifier.verify_code(test_code, test_start, test_goal)
        print(f"  ✅ Verification completed")
        print(f"  ✅ Scores: {scores}")
        print(f"  ✅ Feedback length: {len(feedback)} chars")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70 + "\n")


def test_full_pipeline(pair_id: int = 1):
    """Test full pipeline with real scenes"""
    print("="*70)
    print(f"FULL PIPELINE TEST - Pair {pair_id}")
    print("="*70)
    
    dataset_dir = PROJECT_ROOT / "dataset" / "blender_files"
    start_blend = dataset_dir / f"start_{pair_id:04d}.blend"
    goal_blend = dataset_dir / f"goal_{pair_id:04d}.blend"
    
    if not start_blend.exists() or not goal_blend.exists():
        print(f"❌ Files not found!")
        print(f"   Looking for: {start_blend}")
        print(f"   Looking for: {goal_blend}")
        return None
    
    # Render videos
    print("\n[Step 1] Rendering videos from Blender files...")
    print(f"  Rendering: {start_blend.name}")
    start_video = render_blend_file(str(start_blend), num_frames=30)
    
    print(f"  Rendering: {goal_blend.name}")
    goal_video = render_blend_file(str(goal_blend), num_frames=30)
    
    # Initialize pipeline
    print("\n[Step 2] Initializing pipeline...")
    coder = ZeroShotWorldCoder(
        vjepa_model_path=str(PROJECT_ROOT / 'models' / 'vjepa' / 'vitl16.pth.tar'),
        llm_api_key=GEMINI_API_KEY,
        llm_provider='gemini',
        llm_model='gemini-2.0-flash-exp',
        max_iterations=3
    )
    
    # Run transformation
    print("\n[Step 3] Running transformation...")
    print("  Extracting embeddings...")
    print("  Generating code...")
    print("  Verifying physics...")
    
    try:
        code, scores = coder.transform(start_video, goal_video)
        
        # Save code
        output_path = str(PROJECT_ROOT / 'examples' / f'generated_code_pair_{pair_id}_comprehensive.py')
        with open(output_path, 'w') as f:
            f.write(code)
        
        print("\n" + "="*70)
        print("✅ RESULTS")
        print("="*70)
        print(f"\n📊 Scores:")
        for key, value in scores.items():
            print(f"   {key}: {value:.3f}")
        
        print(f"\n💾 Code saved to: {output_path}")
        print(f"   Length: {len(code)} characters, {len(code.splitlines())} lines")
        
        # Code analysis
        print(f"\n📝 Code Analysis:")
        print(f"   Imports: {'import' in code}")
        print(f"   Physics: {'rigidbody' in code.lower() or 'physics' in code.lower()}")
        print(f"   Objects: {('sphere' in code.lower() or 'cube' in code.lower() or 'mesh' in code.lower())}")
        print(f"   Animation: {'frame' in code.lower()}")
        
        # Syntax check
        try:
            compile(code, '<string>', 'exec')
            print(f"   Syntax: ✅ Valid")
        except SyntaxError as e:
            print(f"   Syntax: ❌ Invalid - {e}")
        
        print(f"\n📄 Code Preview (first 800 chars):")
        print("-"*70)
        print(code[:800])
        if len(code) > 800:
            print("...")
        print("-"*70)
        
        return code, scores
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPREHENSIVE ZERO-SHOT WORLDCODER TEST")
    print("="*70)
    
    # Test components first
    test_individual_components()
    
    # Test full pipeline
    code, scores = test_full_pipeline(pair_id=1)
    
    if code:
        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETE!")
        print("="*70)
        print("\n💡 Next steps:")
        print("   1. Review generated code")
        print("   2. Test in Blender")
        print("   3. Compare with goal scene")
        print("="*70 + "\n")
    else:
        print("\n❌ Tests incomplete. Check errors above.\n")

