"""
Example usage of Zero-Shot WorldCoder
======================================
Shows how to use the pipeline with your generated Blender scenes
"""

import numpy as np
from pathlib import Path
from zero_shot_worldcoder import ZeroShotWorldCoder
import bpy
import cv2

def render_blend_to_video(blend_file: str, output_path: str, frames: int = 300) -> np.ndarray:
    """Render Blender .blend file to video frames"""
    # Open blend file
    bpy.ops.wm.open_mainfile(filepath=blend_file)
    
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    
    # Render frames
    video_frames = []
    for frame in range(1, min(frames + 1, 100), 5):  # Sample frames
        scene.frame_set(frame)
        scene.render.filepath = f"{output_path}_frame_{frame}.png"
        bpy.ops.render.render(write_still=True)
        
        # Load rendered frame
        img = cv2.imread(scene.render.filepath)
        if img is not None:
            video_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return np.array(video_frames)


def example_use_existing_blend_files():
    """Example: Use your generated .blend files"""
    dataset_dir = Path("/Users/akhileshvangala/Desktop/CVPR/dataset/blender_files")
    
    # Load start and goal scenes
    start_blend = dataset_dir / "start_0001.blend"
    goal_blend = dataset_dir / "goal_0001.blend"
    
    print("Loading Blender scenes...")
    start_video = render_blend_to_video(str(start_blend), "/tmp/start")
    goal_video = render_blend_to_video(str(goal_blend), "/tmp/goal")
    
    # Initialize WorldCoder
    coder = ZeroShotWorldCoder(
        # Add your API keys here:
        # llm_api_key=os.getenv("OPENAI_API_KEY"),
        max_iterations=3
    )
    
    # Transform
    code, scores = coder.transform(start_video, goal_video)
    
    # Save generated code
    output_path = Path("/Users/akhileshvangala/Desktop/CVPR/generated_code.py")
    with open(output_path, 'w') as f:
        f.write(code)
    
    print(f"\n✅ Generated code saved to: {output_path}")
    return code, scores


def example_use_video_files():
    """Example: Use video files directly"""
    start_video_path = "/path/to/start_video.mp4"
    goal_video_path = "/path/to/goal_video.mp4"
    
    coder = ZeroShotWorldCoder(max_iterations=3)
    
    # Transform (will load videos internally)
    code, scores = coder.transform(
        start_video=None,  # Will load from path
        goal_video=None,
        start_video_path=start_video_path,
        goal_video_path=goal_video_path
    )
    
    return code, scores


def example_minimal():
    """Minimal example with placeholder videos"""
    # Initialize
    coder = ZeroShotWorldCoder(max_iterations=2)
    
    # Placeholder videos (in practice, load real videos)
    start_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
    goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8)
    
    # Transform
    code, scores = coder.transform(start_video, goal_video)
    
    print("\nGenerated code preview:")
    print(code[:200])
    print("\nScores:", scores)
    
    return code, scores


if __name__ == "__main__":
    print("Zero-Shot WorldCoder Examples")
    print("="*70)
    
    # Run minimal example
    code, scores = example_minimal()
    
    print("\n💡 To use with real videos:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Load V-JEPA model (or use placeholder)")
    print("3. Provide start/goal videos")
    print("4. Run transform()")





