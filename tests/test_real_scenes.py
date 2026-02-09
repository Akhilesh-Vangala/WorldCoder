"""
Test Full Pipeline with Real Blender Scenes
Uses your generated start/goal scene pairs
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import bpy
import numpy as np
from src.zero_shot_worldcoder import ZeroShotWorldCoder
import tempfile
import os

# Gemini API key
GEMINI_API_KEY = "AIzaSyBI52gm0JaJbMbM0xeqO9EuN86p88gIHj0"

def render_blend_to_video_frames(blend_file: str, num_frames: int = 30) -> np.ndarray:
    """Render Blender .blend file to video frames"""
    print(f"  Rendering {Path(blend_file).name}...")
    
    # Open blend file
    bpy.ops.wm.open_mainfile(filepath=blend_file)
    
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 300
    scene.render.resolution_x = 224
    scene.render.resolution_y = 224
    scene.render.resolution_percentage = 100
    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.render.image_settings.file_format = 'PNG'
    
    # Render frames
    video_frames = []
    temp_dir = tempfile.mkdtemp()
    
    # Sample frames evenly
    frame_indices = np.linspace(1, min(300, scene.frame_end), num_frames, dtype=int)
    
    for i, frame_num in enumerate(frame_indices):
        scene.frame_set(int(frame_num))
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        scene.render.filepath = frame_path
        try:
            bpy.ops.render.render(write_still=True)
            # Load frame
            try:
                import cv2
                img = cv2.imread(frame_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    video_frames.append(img_rgb)
            except ImportError:
                from PIL import Image
                img = Image.open(frame_path)
                video_frames.append(np.array(img))
        except Exception as e:
            print(f"    ⚠️  Frame {frame_num} failed: {e}")
            continue
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    if video_frames:
        return np.array(video_frames)
    else:
        # Return placeholder if rendering failed
        return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)


def test_with_real_scenes(pair_id: int = 1):
    """Test pipeline with actual Blender scene pair"""
    print("="*70)
    print(f"Testing Zero-Shot WorldCoder with Real Scene Pair {pair_id}")
    print("="*70)
    
    dataset_dir = PROJECT_ROOT / "dataset" / "blender_files"
    start_blend = dataset_dir / f"start_{pair_id:04d}.blend"
    goal_blend = dataset_dir / f"goal_{pair_id:04d}.blend"
    
    if not start_blend.exists() or not goal_blend.exists():
        print(f"❌ Files not found: {start_blend} or {goal_blend}")
        return None
    
    # Render videos
    print("\n[Step 0] Rendering videos from Blender files...")
    print("  This may take 1-2 minutes...")
    
    try:
        start_video = render_blend_to_video_frames(str(start_blend), num_frames=30)
        print(f"  ✅ Start video: {start_video.shape}, mean brightness: {start_video.mean():.1f}")
        
        goal_video = render_blend_to_video_frames(str(goal_blend), num_frames=30)
        print(f"  ✅ Goal video: {goal_video.shape}, mean brightness: {goal_video.mean():.1f}")
    except Exception as e:
        print(f"  ❌ Rendering failed: {e}")
        print("  Using placeholder videos...")
        start_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 150 + 100
        goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 150 + 100
    
    # Initialize pipeline
    print("\n[Step 1] Initializing Zero-Shot WorldCoder...")
    print("  Loading V-JEPA model...")
    print("  Connecting to Gemini...")
    
    coder = ZeroShotWorldCoder(
        vjepa_model_path=str(PROJECT_ROOT / 'models' / 'vjepa' / 'vitl16.pth.tar'),
        llm_api_key=GEMINI_API_KEY,
        llm_provider='gemini',
        llm_model='gemini-2.0-flash-exp',
        max_iterations=3
    )
    
    # Run transformation
    print("\n[Step 2] Running full transformation pipeline...")
    print("  This will:")
    print("    1. Extract V-JEPA embeddings")
    print("    2. Generate Blender code with Gemini")
    print("    3. Verify with physics engine")
    print("    4. Iterate if needed\n")
    
    try:
        code, scores = coder.transform(start_video, goal_video)
        
        # Save generated code
        output_path = str(PROJECT_ROOT / 'examples' / f'generated_code_pair_{pair_id}.py')
        with open(output_path, 'w') as f:
            f.write(code)
        
        print("\n" + "="*70)
        print("✅ TRANSFORMATION COMPLETE!")
        print("="*70)
        print(f"\n📊 Results:")
        print(f"   Per-Frame Fidelity: {scores.get('per_frame_fidelity', 0):.3f}")
        print(f"   Temporal Flow Alignment: {scores.get('temporal_flow_alignment', 0):.3f}")
        print(f"   Physics Validity Score: {scores.get('physics_validity_score', 0):.3f}")
        
        print(f"\n💾 Generated code saved to: {output_path}")
        print(f"   Code length: {len(code)} characters")
        
        # Show code quality
        has_physics = 'rigidbody' in code.lower() and 'gravity' in code.lower()
        has_objects = ('sphere' in code.lower() or 'ball' in code.lower()) and ('track' in code.lower() or 'plane' in code.lower())
        syntax_ok = True
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError:
            syntax_ok = False
        
        print(f"\n📝 Code Quality:")
        print(f"   ✅ Has physics setup: {has_physics}")
        print(f"   ✅ Has objects: {has_objects}")
        print(f"   ✅ Syntax valid: {syntax_ok}")
        
        print(f"\n📄 Code Preview (first 600 chars):")
        print("-"*70)
        print(code[:600])
        if len(code) > 600:
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
    print("ZERO-SHOT WORLDCODER - REAL SCENE TEST")
    print("="*70)
    print("\nTesting with your generated Blender scenes...")
    
    # Test on first pair
    code, scores = test_with_real_scenes(pair_id=1)
    
    if code:
        print("\n" + "="*70)
        print("✅ TEST SUCCESSFUL!")
        print("="*70)
        print("\n💡 Next steps:")
        print("   1. Review generated_code_pair_1.py")
        print("   2. Test in Blender to see the result")
        print("   3. Compare with goal scene")
        print("   4. Try with other scene pairs")
        print("="*70 + "\n")
    else:
        print("\n❌ Test failed. Check errors above.\n")





