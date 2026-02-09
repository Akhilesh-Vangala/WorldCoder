"""
Test Zero-Shot WorldCoder with actual Blender scenes
Renders videos from .blend files and tests the pipeline
"""

import bpy
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '/Users/akhileshvangala/Desktop/CVPR')

import numpy as np
from zero_shot_worldcoder import ZeroShotWorldCoder
import tempfile

def render_blend_to_video_frames(blend_file: str, num_frames: int = 30, 
                                  resolution: tuple = (224, 224)) -> np.ndarray:
    """
    Render Blender .blend file to video frames
    
    Returns:
        video: (T, H, W, C) numpy array
    """
    print(f"  Rendering {blend_file}...")
    
    # Open blend file
    bpy.ops.wm.open_mainfile(filepath=blend_file)
    
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 300
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
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
        bpy.ops.render.render(write_still=True)
        
        # Load frame
        try:
            import cv2
            img = cv2.imread(frame_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video_frames.append(img_rgb)
        except ImportError:
            # Fallback: use PIL if cv2 not available
            from PIL import Image
            img = Image.open(frame_path)
            video_frames.append(np.array(img))
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    if video_frames:
        return np.array(video_frames)
    else:
        # Return placeholder if rendering failed
        return np.zeros((num_frames, resolution[0], resolution[1], 3), dtype=np.uint8)


def test_with_blender_scenes(pair_id: int = 1):
    """Test pipeline with actual Blender scene pair"""
    dataset_dir = Path("/Users/akhileshvangala/Desktop/CVPR/dataset/blender_files")
    
    start_blend = dataset_dir / f"start_{pair_id:04d}.blend"
    goal_blend = dataset_dir / f"goal_{pair_id:04d}.blend"
    
    if not start_blend.exists() or not goal_blend.exists():
        print(f"❌ Files not found: {start_blend} or {goal_blend}")
        return None
    
    print("="*70)
    print(f"Testing Zero-Shot WorldCoder on Pair {pair_id}")
    print("="*70)
    
    # Render videos
    print("\n[Step 0] Rendering videos from Blender files...")
    print("  This may take a minute...")
    
    try:
        start_video = render_blend_to_video_frames(str(start_blend), num_frames=30)
        print(f"  ✅ Start video: {start_video.shape}")
        
        goal_video = render_blend_to_video_frames(str(goal_blend), num_frames=30)
        print(f"  ✅ Goal video: {goal_video.shape}")
    except Exception as e:
        print(f"  ❌ Rendering failed: {e}")
        print("  Using placeholder videos...")
        start_video = np.zeros((30, 224, 224, 3), dtype=np.uint8)
        goal_video = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    
    # Initialize pipeline
    print("\n[Step 1] Initializing Zero-Shot WorldCoder...")
    coder = ZeroShotWorldCoder(max_iterations=2)  # Start with 2 iterations for testing
    
    # Test transformation
    print("\n[Step 2] Running transformation pipeline...")
    try:
        code, scores = coder.transform(start_video, goal_video)
        
        print("\n" + "="*70)
        print("✅ TEST COMPLETE")
        print("="*70)
        print(f"\nGenerated Code Length: {len(code)} characters")
        print(f"\nScores:")
        for key, value in scores.items():
            print(f"  {key}: {value:.3f}")
        
        # Save generated code
        output_path = Path(f"/Users/akhileshvangala/Desktop/CVPR/generated_code_pair_{pair_id}.py")
        with open(output_path, 'w') as f:
            f.write(code)
        print(f"\n💾 Generated code saved to: {output_path}")
        
        # Show code preview
        print("\n📝 Code Preview (first 500 chars):")
        print("-"*70)
        print(code[:500])
        if len(code) > 500:
            print("...")
        print("-"*70)
        
        return code, scores
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_code_quality(code: str) -> dict:
    """Check if generated code is syntactically correct and has key components"""
    checks = {
        'has_bpy_import': 'import bpy' in code or 'from bpy import' in code,
        'has_physics_setup': 'rigidbody' in code.lower() or 'physics' in code.lower(),
        'has_gravity': 'gravity' in code.lower(),
        'has_object_creation': ('sphere' in code.lower() or 'cube' in code.lower() or 
                                'mesh' in code.lower()),
        'has_friction': 'friction' in code.lower(),
        'syntax_valid': False,
    }
    
    # Check syntax
    try:
        compile(code, '<string>', 'exec')
        checks['syntax_valid'] = True
    except SyntaxError:
        checks['syntax_valid'] = False
    
    return checks


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Zero-Shot WorldCoder - Video Test")
    print("="*70)
    
    # Test on first pair
    code, scores = test_with_blender_scenes(pair_id=1)
    
    if code:
        print("\n" + "="*70)
        print("Code Quality Check")
        print("="*70)
        
        quality = test_code_quality(code)
        print("\n✅ Quality Checks:")
        for check, passed in quality.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {passed}")
        
        overall = sum(quality.values()) / len(quality)
        print(f"\n📊 Overall Quality Score: {overall*100:.1f}%")
        
        if overall >= 0.8:
            print("\n✅ Code looks good! Ready to use.")
        elif overall >= 0.5:
            print("\n⚠️  Code has some issues but might work.")
        else:
            print("\n❌ Code needs improvement.")
    
    print("\n" + "="*70)

