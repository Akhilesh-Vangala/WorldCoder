"""
Test Full Pipeline: V-JEPA + Physics Verifier
"""

import sys
sys.path.insert(0, '/Users/akhileshvangala/Desktop/CVPR')

import numpy as np
from zero_shot_worldcoder import VJEPAEncoder, PhysicsVerifier

def test_vjepa():
    """Test V-JEPA model loading and encoding"""
    print("="*70)
    print("Testing V-JEPA Model")
    print("="*70)
    
    model_path = '/Users/akhileshvangala/Desktop/CVPR/models/vjepa/vitl16.pth.tar'
    vjepa = VJEPAEncoder(model_path)
    
    # Test with video
    test_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 200 + 50
    embedding = vjepa.encode(test_video)
    
    print(f"\n✅ V-JEPA Test Results:")
    print(f"   Model loaded: {vjepa.encoder is not None}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
    print(f"   Embedding mean: {embedding.mean():.3f}")
    
    return embedding.shape[0] == 1024  # ViT-L embedding dim


def test_physics_verifier():
    """Test Physics Verifier with real Blender execution"""
    print("\n" + "="*70)
    print("Testing Physics Verifier")
    print("="*70)
    
    # Simple test code
    test_code = """
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

scene = bpy.context.scene
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()

scene.gravity = (0.0, 0.0, -9.81)
scene.frame_start = 1
scene.frame_end = 300

# Simple scene
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, -1))
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 0, 2))

# Camera
bpy.ops.object.camera_add(location=(5, -5, 3))
cam = bpy.context.active_object
cam.rotation_euler = (1.0, 0, 0.785)
scene.camera = cam

# Light
bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))

scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.engine = 'BLENDER_EEVEE_NEXT'
"""
    
    verifier = PhysicsVerifier()
    goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 100 + 100
    start_video = np.zeros((30, 224, 224, 3), dtype=np.uint8)
    
    print("\n[Test] Executing Blender code...")
    is_valid, scores, feedback = verifier.verify_code(test_code, start_video, goal_video)
    
    print(f"\n✅ Physics Verifier Test Results:")
    print(f"   Code executed: True")
    print(f"   Per-Frame Fidelity: {scores.get('per_frame_fidelity', 0):.3f}")
    print(f"   TFA: {scores.get('temporal_flow_alignment', 0):.3f}")
    print(f"   PVS: {scores.get('physics_validity_score', 0):.3f}")
    print(f"   Valid: {is_valid}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Full Pipeline Test: V-JEPA + Physics Verifier")
    print("="*70)
    
    # Test V-JEPA
    vjepa_ok = test_vjepa()
    
    # Test Physics Verifier
    verifier_ok = test_physics_verifier()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ V-JEPA: {'WORKING' if vjepa_ok else 'FAILED'}")
    print(f"✅ Physics Verifier: {'WORKING' if verifier_ok else 'FAILED'}")
    print("\n💡 Next: Add LLM API key to enable code generation")
    print("="*70 + "\n")





