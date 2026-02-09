import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 1. Set up physics world with gravity
bpy.context.scene.gravity = (0, 0, -9.81)

# 2. Create a track with appropriate geometry
# Create a plane for the track
bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, 0))
track = bpy.context.object
track.name = "Track"

# Rotate the track to create a ramp
ramp_angle = 20.0  # Gentler ramp due to slower motion
track.rotation_euler = (math.radians(ramp_angle), 0, 0)

# Add collision physics to the track
track.select_set(True)
bpy.context.view_layer.objects.active = track
bpy.ops.object.modifier_add(type='COLLISION')
track.modifiers["Collision"].friction = 0.35 # Higher friction due to slower motion
track.modifiers["Collision"].restitution = 0.1

# 3. Create a ball with correct physics properties
# Create a sphere for the ball
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, enter_editmode=False, align='WORLD', location=(0, 2, 2))
ball = bpy.context.object
ball.name = "Ball"

# Add rigid body physics to the ball
bpy.ops.rigidbody.object_add(type='ACTIVE')
ball.rigid_body.mass = 1.5 # Adjust mass
ball.rigid_body.friction = 0.35 # Higher friction due to slower motion
ball.rigid_body.restitution = 0.1 # Adjust restitution
ball.rigid_body.use_margin = True
ball.rigid_body.collision_margin = 0.01

# 4. Use parameters inferred from the embedding analysis
# (Parameters are already set above based on the analysis)

# Optionally, set the starting frame and end frame for the animation
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 250

# Bake the physics simulation (optional, but useful for playback)
bpy.ops.object.select_all(action='DESELECT')
ball.select_set(True)
bpy.context.view_layer.objects.active = ball
bpy.ops.object.bake_all_physics(bake=True)