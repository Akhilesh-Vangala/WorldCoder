
import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

scene = bpy.context.scene
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()

scene.gravity = (0.0, 0.0, -9.81)
scene.frame_start = 1
scene.frame_end = 300

# Add track and ball here
