import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Physics parameters (inferred from embedding analysis)
ball_mass = 1.2
ramp_angle = 15.0  # Gentler ramp for slower, upward motion
track_friction = 0.35 # Higher friction for slower motion
ball_friction = 0.35
restitution = 0.15 # Lower restitution for less bouncing

# --- Scene Setup ---

# Gravity settings (optional, but good to be explicit)
bpy.context.scene.gravity = (0, 0, -9.81)

# --- Track Creation ---

# Create a plane for the track
bpy.ops.mesh.primitive_plane_add(size=5, enter_editmode=False, align='WORLD', location=(0, 0, 0))
track = bpy.context.object
track.name = "Track"

# Rotate the track to create a ramp
track.rotation_euler[0] = math.radians(ramp_angle)

# Add collision physics to the track
track.modifiers.new(name="Collision_Mesh", type='COLLISION')
track.collision.use_margin = True
track.collision.friction = track_friction
track.collision.restitution = 0.01 # Small value for track

# --- Ball Creation ---

# Create a sphere for the ball
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, enter_editmode=False, align='WORLD', location=(0, 0.5, 0.5))
ball = bpy.context.object
ball.name = "Ball"

# Add rigid body physics to the ball
bpy.ops.rigidbody.object_add(type='ACTIVE')
ball.rigid_body.mass = ball_mass
ball.rigid_body.friction = ball_friction
ball.rigid_body.restitution = restitution
ball.rigid_body.use_margin = True
ball.rigid_body.collision_shape = 'SPHERE'

# Initial velocity (upward/forward) - adjust as needed
ball.rigid_body.initial_linear_velocity = (0.5, 0, 0.2)  # X, Y, Z velocity

# --- Animation Settings ---

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 250
bpy.context.scene.render.fps = 24

# --- Bake Physics (Optional) ---
# bpy.ops.object.select_all(action='DESELECT')
# ball.select_set(True)
# bpy.context.view_layer.objects.active = ball
# bpy.ops.object.bake_all_physics(bake=True)