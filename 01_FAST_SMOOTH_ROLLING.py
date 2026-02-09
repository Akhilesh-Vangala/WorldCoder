"""
SCENARIO 1: FAST SMOOTH ROLLING
================================

PHYSICS CONCEPTS:
- Low friction coefficient (μ = 0.2)
- Steep inclines maximize gravitational acceleration
- Light mass reduces rolling resistance
- Low restitution prevents energy loss to bouncing

EXPECTED BEHAVIOR:
- Ball accelerates quickly on tilted surface
- High velocity down the ramp
- Smooth continuous rolling
- Travels far on final surface before stopping

KEY FORMULAS:
- Acceleration: a = (5/7)g sin(θ) ≈ 0.714 × 9.81 × sin(35°) ≈ 4.0 m/s²
- Final velocity: v = √(2gh × 7/5) where h is total height dropped
- Friction force: f = μN = 0.2 × mg cos(θ)

REAL-WORLD ANALOGY:
- Steel ball bearing on polished steel track
- Bowling ball on oiled lane
"""

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
scene.frame_end = 500
scene.render.fps = 30

print("=" * 70)
print("SCENARIO 1: FAST SMOOTH ROLLING")
print("=" * 70)

# ========== PHYSICS PARAMETERS ==========
# Track geometry
s1_length = 3.0
s1_z = 0.0
ramp_length = 4.0
ramp_angle = 35.0          # STEEP! More acceleration
s3_length = 3.5
step_height = 0.8
s4_length = 4.0
width = 1.5
thickness = 0.08

# Ball properties
ball_radius = 0.20
ball_mass = 1.5            # LIGHT! Less inertia to overcome

# Motion parameters
track_tilt = 8.0           # STEEP START! High initial acceleration

# Build track
def build_track():
    W = width
    ang = radians(ramp_angle)
    tilt = radians(track_tilt)
    halfW = W / 2
    
    x0, z0 = 0.0, s1_z
    x1 = s1_length
    z1 = s1_z - s1_length * sin(tilt)
    x2, z2 = x1 + ramp_length * cos(ang), z1 - ramp_length * sin(ang)
    x3, z3 = x2 + s3_length, z2
    x4, z4 = x3, z3 - step_height
    x5, z5 = x4 + s4_length, z4
    
    print(f"Physics Analysis:")
    print(f"  Initial tilt: {track_tilt}° → a = {0.714*9.81*sin(tilt):.2f} m/s²")
    print(f"  Ramp angle: {ramp_angle}° → a = {0.714*9.81*sin(ang):.2f} m/s²")
    print(f"  Total height drop: {z0 - z5:.2f} m")
    print(f"  Potential energy: {ball_mass * 9.81 * (z0 - z5):.2f} J")
    
    me = bpy.data.meshes.new("TrackMesh")
    obj = bpy.data.objects.new("Track", me)
    bm = bmesh.new()
    
    pts = [
        (x0, -halfW, z0), (x0, halfW, z0),
        (x1, -halfW, z1), (x1, halfW, z1),
        (x2, -halfW, z2), (x2, halfW, z2),
        (x3, -halfW, z3), (x3, halfW, z3),
        (x3, -halfW, z4), (x3, halfW, z4),
        (x5, -halfW, z5), (x5, halfW, z5),
    ]
    
    v = [bm.verts.new(p) for p in pts]
    bm.verts.ensure_lookup_table()
    
    bm.faces.new([v[0], v[1], v[3], v[2]])
    bm.faces.new([v[2], v[3], v[5], v[4]])
    bm.faces.new([v[4], v[5], v[7], v[6]])
    bm.faces.new([v[6], v[7], v[9], v[8]])
    bm.faces.new([v[8], v[9], v[11], v[10]])
    
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    
    solid = obj.modifiers.new("Solidify", 'SOLIDIFY')
    solid.thickness = thickness
    solid.offset = -1.0
    
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    
    bpy.ops.rigidbody.object_add(type='PASSIVE')
    rb = obj.rigid_body
    rb.collision_shape = 'MESH'
    rb.friction = 0.2          # LOW FRICTION! Smooth surface
    rb.restitution = 0.05      # Minimal bounce
    rb.collision_margin = 0.001
    rb.mesh_source = 'FINAL'
    
    print(f"\n  Track friction: μ = {rb.friction} (ice-like)")
    
    mat = bpy.data.materials.new(name="TrackMat")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.7, 0.8, 0.9, 1.0)
    mat.node_tree.nodes["Principled BSDF"].inputs['Metallic'].default_value = 0.8
    mat.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.1
    obj.data.materials.append(mat)
    
    return obj, x0, z0

track, start_x, start_z = build_track()

# Create ball
ball_x = start_x + 0.3
ball_y = 0.0
ball_z = start_z + thickness/2 + ball_radius + 0.3

bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_radius, location=(ball_x, ball_y, ball_z))
ball = bpy.context.active_object
ball.name = "Ball"

bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

bpy.ops.rigidbody.object_add(type='ACTIVE')
rb = ball.rigid_body
rb.type = 'ACTIVE'
rb.collision_shape = 'SPHERE'
rb.mass = ball_mass
rb.friction = 0.2          # LOW FRICTION! Reduces rolling resistance
rb.restitution = 0.05      # NO BOUNCE! All energy into forward motion
rb.linear_damping = 0.01   # Low air resistance
rb.angular_damping = 0.01  # Maintains spin
rb.use_deactivation = False
rb.kinematic = False
rb.enabled = True

print(f"  Ball friction: μ = {rb.friction}")
print(f"  Ball mass: {ball_mass} kg")
print(f"  Restitution: e = {rb.restitution} (minimal bounce)")

# Ball material (blue - fast and cool!)
mat_ball = bpy.data.materials.new(name="BallMat")
mat_ball.use_nodes = True
mat_ball.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.1, 0.3, 1.0, 1.0)
mat_ball.node_tree.nodes["Principled BSDF"].inputs['Metallic'].default_value = 0.8
mat_ball.node_tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.2
ball.data.materials.append(mat_ball)

# Camera
cam_x = 7.0
cam_y = -12.0
cam_z = 6.0

bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
cam = bpy.context.active_object
cam.rotation_euler = (radians(70), 0, 0)
scene.camera = cam

# Lighting
bpy.ops.object.light_add(type='SUN', location=(8, -8, 20))
sun = bpy.context.active_object
sun.data.energy = 6.0
sun.rotation_euler = (radians(50), 0, radians(30))

scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.engine = 'BLENDER_EEVEE_NEXT'

scene.frame_set(1)

# Save
output_path = "/Users/akhileshvangala/Desktop/cvpr/scenarios/01_FAST_SMOOTH_ROLLING.blend"
bpy.ops.wm.save_as_mainfile(filepath=output_path)

print("\n" + "=" * 70)
print("✅ SCENARIO 1: FAST SMOOTH ROLLING - Ready!")
print("=" * 70)
print("OBSERVE:")
print("  • Ball accelerates rapidly on steep tilt")
print("  • High speed down 35° ramp")
print("  • Smooth continuous motion (no jerks)")
print("  • Travels far on final surface")
print(f"\nSaved: {output_path}")
print("=" * 70)


