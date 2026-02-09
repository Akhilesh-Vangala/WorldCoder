"""
Test Physics Verifier with real Blender code execution
"""

import sys
sys.path.insert(0, '/Users/akhileshvangala/Desktop/CVPR')

import numpy as np
from zero_shot_worldcoder import PhysicsVerifier
from pathlib import Path

def test_verifier():
    """Test physics verifier with actual Blender scene code"""
    
    # Load your existing scene code as test
    test_code = """
import bpy, bmesh
from math import radians, cos, sin

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

scene = bpy.context.scene

# Physics world
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()

scene.rigidbody_world.time_scale = 1.0
scene.rigidbody_world.substeps_per_frame = 60
scene.rigidbody_world.solver_iterations = 30
scene.gravity = (0.0, 0.0, -9.81)
scene.frame_start = 1
scene.frame_end = 300
scene.render.fps = 30

# Simple track
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
track = bpy.context.active_object
track.scale = (4, 1, 0.1)
track.rotation_euler = (radians(25), 0, 0)

bpy.ops.rigidbody.object_add(type='PASSIVE')
track.rigid_body.friction = 0.2

# Ball
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0, 0, 1))
ball = bpy.context.active_object
ball.name = "Ball"

bpy.ops.rigidbody.object_add(type='ACTIVE')
ball.rigid_body.mass = 2.0
ball.rigid_body.friction = 0.2
ball.rigid_body.restitution = 0.05

# Camera
bpy.ops.object.camera_add(location=(5, -8, 4))
cam = bpy.context.active_object
cam.rotation_euler = (radians(60), 0, 0)
scene.camera = cam

# Lighting
bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
sun = bpy.context.active_object
sun.data.energy = 6.0

scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.engine = 'BLENDER_EEVEE_NEXT'
"""
    
    print("="*70)
    print("Testing Physics Verifier with Real Blender Code")
    print("="*70)
    
    verifier = PhysicsVerifier()
    
    # Create dummy goal video (for comparison)
    goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 100 + 100  # Grayish
    start_video = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    
    print("\n[Test] Executing Blender code and rendering...")
    is_valid, scores, feedback = verifier.verify_code(test_code, start_video, goal_video)
    
    print(f"\n✅ Verification complete!")
    print(f"   Valid: {is_valid}")
    print(f"   Scores: {scores}")
    print(f"   Feedback: {feedback}")
    
    return is_valid, scores, feedback


if __name__ == "__main__":
    test_verifier()

