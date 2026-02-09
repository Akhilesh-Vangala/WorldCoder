"""
Zero-Shot WorldCoder: V-JEPA + LLM + Physics Verifier
========================================================
No dataset needed - pure zero-shot approach with self-correcting physics loop
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # Placeholder

import numpy as np
import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Dict, List, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Placeholder for V-JEPA - you'll need to load actual model
try:
    import vjepa  # You'll need to install/load V-JEPA model
    VJEPA_AVAILABLE = True
except:
    VJEPA_AVAILABLE = False
    print("⚠️  V-JEPA not loaded. Using placeholder embeddings.")

# LLM client - support for Gemini and OpenAI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False


class VJEPAEncoder:
    """Wrapper for V-JEPA embedding extraction"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.encoder = None
        self.embedding_dim = 1024  # ViT-L default embedding dimension
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
        
        if model_path:
            self._load_model(model_path)
        else:
            # Try to load from environment or default paths
            self._try_auto_load()
    
    def _try_auto_load(self):
        """Try to automatically find and load V-JEPA model"""
        # Check common locations
        possible_paths = [
            os.getenv('VJEPA_MODEL_PATH'),
            '/Users/akhileshvangala/Desktop/CVPR/models/vjepa/vitl16.pth.tar',
            'models/vjepa/vitl16.pth.tar',
            '~/.vjepa/vitl16.pth.tar',
        ]
        
        for path in possible_paths:
            if path and os.path.exists(os.path.expanduser(path)):
                self._load_model(os.path.expanduser(path))
                return
        
        print("📝 V-JEPA model not found. Using placeholder embeddings.")
        print("   To load V-JEPA: Set VJEPA_MODEL_PATH or pass model_path to VJEPAEncoder()")
        print("   Default path checked: /Users/akhileshvangala/Desktop/CVPR/models/vjepa/vitl16.pth.tar")
    
    def _load_model(self, path: str):
        """Load pre-trained V-JEPA model"""
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available. Cannot load V-JEPA model.")
            return
        
        try:
            expanded_path = os.path.expanduser(path)
            if not os.path.exists(expanded_path):
                print(f"⚠️  Model file not found: {expanded_path}")
                return
            
            print(f"📥 Loading V-JEPA model from {expanded_path}...")
            
            # Add jepa to path if needed
            jepa_path = '/Users/akhileshvangala/Desktop/CVPR/jepa'
            if jepa_path not in sys.path:
                sys.path.insert(0, jepa_path)
            
            # Import V-JEPA modules
            try:
                import src.models.vision_transformer as video_vit
                from src.models.utils.multimask import MultiMaskWrapper
            except ImportError:
                print("⚠️  Cannot import V-JEPA modules. Install dependencies: pip install -r jepa/requirements.txt")
                return
            
            # Load checkpoint
            checkpoint = torch.load(expanded_path, map_location=self.device)
            
            # Initialize model architecture (ViT-L for vitl16.pth.tar)
            model_name = 'vit_large'  # ViT-L model
            encoder = video_vit.__dict__[model_name](
                img_size=224,
                patch_size=16,
                num_frames=16,  # V-JEPA uses 16 frames
                tubelet_size=2,
                uniform_power=False,
            )
            
            # Wrap in MultiMaskWrapper (V-JEPA architecture)
            encoder = MultiMaskWrapper(encoder)
            
            # Load weights from checkpoint
            checkpoint_key = 'target_encoder' if 'target_encoder' in checkpoint else 'encoder'
            if checkpoint_key in checkpoint:
                state_dict = checkpoint[checkpoint_key]
                # Remove 'backbone.' prefix if present (MultiMaskWrapper adds it)
                if any(k.startswith('backbone.') for k in state_dict.keys()):
                    # Already has backbone prefix
                    encoder.load_state_dict(state_dict, strict=False)
                else:
                    # Add backbone prefix for MultiMaskWrapper
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        new_state_dict['backbone.' + k] = v
                    encoder.load_state_dict(new_state_dict, strict=False)
            elif 'state_dict' in checkpoint:
                encoder.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                encoder.load_state_dict(checkpoint, strict=False)
            
            encoder.to(self.device)
            encoder.eval()
            
            self.encoder = encoder
            self.embedding_dim = encoder.backbone.embed_dim  # 1024 for ViT-L
            
            print(f"✅ V-JEPA model loaded successfully!")
            print(f"   Model: {model_name}, Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            print(f"⚠️  Failed to load V-JEPA model: {e}")
            import traceback
            traceback.print_exc()
            print("   Using placeholder embeddings.")
    
    def _encode_with_model(self, video_tensor) -> np.ndarray:
        """Encode video using loaded V-JEPA model"""
        if self.encoder is None:
            raise ValueError("Model not loaded")
        
        # V-JEPA expects (B, C, T, H, W) format for video
        # Current: (T, C, H, W) or (B, T, C, H, W)
        if video_tensor.dim() == 4:  # (T, C, H, W)
            video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
        
        # Convert to (B, C, T, H, W)
        if video_tensor.dim() == 5 and video_tensor.shape[1] == 3:  # (B, C, T, H, W) already
            pass
        elif video_tensor.dim() == 5:  # (B, T, C, H, W)
            video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # V-JEPA uses 16 frames, pad or sample if needed
        B, C, T, H, W = video_tensor.shape
        target_frames = 16
        
        if T < target_frames:
            # Repeat last frame to pad
            padding = video_tensor[:, :, -1:, :, :].repeat(1, 1, target_frames - T, 1, 1)
            video_tensor = torch.cat([video_tensor, padding], dim=2)
        elif T > target_frames:
            # Sample evenly
            indices = torch.linspace(0, T - 1, target_frames).long()
            video_tensor = video_tensor[:, :, indices, :, :]
        
        # Forward through encoder (no masks needed for encoding)
        with torch.no_grad():
            # Get backbone output (MultiMaskWrapper has .backbone)
            output = self.encoder.backbone(video_tensor, masks=None)
            
            # Output is (B, N, D) where N is number of tokens
            # Global pool: use CLS token or mean pool
            if output.shape[1] > 0:
                # Mean pool over spatial-temporal tokens
                embedding = output.mean(dim=1)  # (B, D)
            else:
                embedding = output.squeeze(1)  # Fallback
            
            # Remove batch dimension
            if embedding.dim() > 1:
                embedding = embedding[0]  # (D,)
            
            return embedding.cpu().numpy()
    
    def encode(self, video: np.ndarray) -> np.ndarray:
        """
        Extract temporal embeddings from video
        
        Args:
            video: (T, H, W, C) numpy array of video frames
        
        Returns:
            embedding: (embedding_dim,) numpy array
        """
        if self.encoder and TORCH_AVAILABLE:
            # Real V-JEPA encoding
            try:
                with torch.no_grad():
                    # Ensure RGB (remove alpha channel if present)
                    if video.ndim == 4 and video.shape[-1] == 4:
                        video = video[:, :, :, :3]  # Remove alpha
                    
                    # Convert to tensor and normalize
                    # V-JEPA typically expects: (B, T, C, H, W), normalized to [0, 1]
                    video_normalized = video.astype(np.float32) / 255.0
                    
                    # Resize to exactly 224x224 (required by V-JEPA)
                    if video_normalized.shape[1] != 224 or video_normalized.shape[2] != 224:
                        import cv2
                        resized_frames = []
                        for frame in video_normalized:
                            # Convert to uint8 for cv2
                            frame_uint8 = (frame * 255).astype(np.uint8)
                            frame_resized = cv2.resize(frame_uint8, (224, 224), interpolation=cv2.INTER_LINEAR)
                            resized_frames.append(frame_resized.astype(np.float32) / 255.0)
                        video_normalized = np.array(resized_frames)
                    
                    # Reshape: (T, H, W, C) -> (T, C, H, W)
                    if video_normalized.ndim == 4 and video_normalized.shape[-1] == 3:
                        video_normalized = np.transpose(video_normalized, (0, 3, 1, 2))
                    
                    video_tensor = torch.from_numpy(video_normalized).float().to(self.device)
                    
                    # Encode
                    embedding = self._encode_with_model(video_tensor)
                    
                    # _encode_with_model already returns numpy array
                    # Flatten if needed (might be (B, D) or (B, T, D))
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()[:self.embedding_dim]
                    
                    return embedding
            except Exception as e:
                print(f"⚠️  V-JEPA encoding failed: {e}")
                print("   Falling back to placeholder embeddings.")
                # Fall through to placeholder
        else:
            # Placeholder: return embedding with realistic structure based on video content
            embedding_dim = self.embedding_dim
            
            # Analyze video to create meaningful placeholder
            # Motion magnitude
            if video.ndim == 4 and video.shape[0] > 1:
                frame_diffs = np.diff(video.astype(np.float32), axis=0)
                motion_magnitude = np.mean(np.abs(frame_diffs))
                
                # Create structured embedding
                embedding = np.random.randn(embedding_dim) * 0.1
                
                # First dimensions: motion intensity
                motion_idx = min(50, embedding_dim // 4)
                embedding[:motion_idx] += motion_magnitude * 0.01
                
                # Middle dimensions: temporal consistency
                consistency = 1.0 / (1.0 + np.std(frame_diffs))
                consistency_idx = min(100, embedding_dim // 2)
                embedding[motion_idx:consistency_idx] += consistency * 0.5
                
                # Gravity/direction (if motion detected)
                if motion_magnitude > 0:
                    # Check if mostly downward motion (typical for rolling balls)
                    avg_motion = np.mean(frame_diffs, axis=(1, 2, 3))
                    if len(avg_motion) > 0 and np.mean(avg_motion) < 0:
                        embedding[consistency_idx:consistency_idx+50] -= 0.3  # Downward bias
                
                return embedding
            else:
                # Fallback: random but structured
                return np.random.randn(embedding_dim) * 0.1


class LLMCodeGenerator:
    """LLM-based code generator with few-shot prompting (supports Gemini and OpenAI)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp", provider: str = "gemini"):
        self.client = None
        self.model_name = model
        self.provider = provider.lower()  # "gemini" or "openai"
        
        if api_key:
            if self.provider == "gemini" and GEMINI_AVAILABLE:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model_name=model)
                print(f"✅ Gemini client initialized with model: {model}")
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                self.client = OpenAI(api_key=api_key)
                print(f"✅ OpenAI client initialized with model: {model}")
            else:
                print(f"⚠️  Provider '{provider}' not available. Install: pip install google-generativeai (Gemini) or openai (OpenAI)")
        else:
            print("📝 LLM client not initialized. Provide api_key.")
    
    def _few_shot_examples(self) -> str:
        """Hand-crafted few-shot examples for prompting"""
        return """
        Example 1:
        Embedding pattern: Fast continuous downward motion, smooth trajectory
        Physics inferred: Low friction (0.1-0.3), gravity-consistent, ramp angle 30-40°
        Code generated:
        ```python
        import bpy
        import math
        
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        scene = bpy.context.scene
        if not scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()
        scene.gravity = (0, 0, -9.81)
        
        # Track with PASSIVE rigid body
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
        track = bpy.context.object
        track.rotation_euler = (math.radians(35.0), 0, 0)
        bpy.ops.rigidbody.object_add(type='PASSIVE')
        track.rigid_body.friction = 0.2  # CORRECT: rigid_body.friction
        
        # Ball with ACTIVE rigid body
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 2, 1))
        ball = bpy.context.active_object
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        ball.rigid_body.mass = 2.0
        ball.rigid_body.friction = 0.2
        ball.rigid_body.restitution = 0.05
        
        # Camera
        bpy.ops.object.camera_add(location=(7.0, -12.0, 6.0))
        cam = bpy.context.active_object
        cam.rotation_euler = (math.radians(70), 0, 0)
        scene.camera = cam
        
        # Lighting
        bpy.ops.object.light_add(type='SUN', location=(8, -8, 20))
        sun = bpy.context.active_object
        sun.data.energy = 6.0
        ```

        Example 2:
        Embedding pattern: Slower motion, more resistance
        Physics inferred: Higher friction (0.3-0.5), gentler ramp
        Code generated:
        ```python
        import bpy
        import math
        
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        scene = bpy.context.scene
        if not scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()
        scene.gravity = (0, 0, -9.81)
        
        # Track
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
        track = bpy.context.object
        track.rotation_euler = (math.radians(25.0), 0, 0)
        bpy.ops.rigidbody.object_add(type='PASSIVE')
        track.rigid_body.friction = 0.4  # CORRECT: use rigid_body
        
        # Ball
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 2, 1))
        ball = bpy.context.object
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        ball.rigid_body.mass = 1.5
        ball.rigid_body.friction = 0.4
        ball.rigid_body.restitution = 0.1
        
        # Camera
        bpy.ops.object.camera_add(location=(6.0, -11.0, 5.0))
        cam = bpy.context.active_object
        cam.rotation_euler = (math.radians(65), 0, 0)
        scene.camera = cam
        
        # Lighting
        bpy.ops.object.light_add(type='SUN', location=(7, -7, 18))
        sun = bpy.context.active_object
        sun.data.energy = 6.0
        ```

        Example 3:
        Embedding pattern: Bouncing, discontinuous motion
        Physics inferred: High restitution (0.3-0.7), lower friction
        Code generated:
        ```python
        import bpy
        import math
        
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        scene = bpy.context.scene
        if not scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()
        scene.gravity = (0, 0, -9.81)
        
        # Track
        bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
        track = bpy.context.object
        track.rotation_euler = (math.radians(30.0), 0, 0)
        bpy.ops.rigidbody.object_add(type='PASSIVE')
        track.rigid_body.friction = 0.15  # CORRECT: rigid_body, NOT modifiers
        
        # Ball
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 2, 1))
        ball = bpy.context.object
        bpy.ops.rigidbody.object_add(type='ACTIVE')
        ball.rigid_body.mass = 1.0
        ball.rigid_body.friction = 0.15
        ball.rigid_body.restitution = 0.5
        
        # Camera
        bpy.ops.object.camera_add(location=(8.0, -13.0, 7.0))
        cam = bpy.context.active_object
        cam.rotation_euler = (math.radians(75), 0, 0)
        scene.camera = cam
        
        # Lighting
        bpy.ops.object.light_add(type='SUN', location=(9, -9, 22))
        sun = bpy.context.active_object
        sun.data.energy = 6.0
        ```
        """
    
    def _analyze_embeddings(self, start_emb: np.ndarray, goal_emb: np.ndarray) -> str:
        """Analyze embeddings to extract motion patterns"""
        diff = goal_emb - start_emb
        
        # Simple heuristics (in practice, could use more sophisticated analysis)
        motion_magnitude = np.linalg.norm(diff)
        direction = np.mean(diff[:100])  # First 100 dims often encode motion
        
        analysis = f"""
        Embedding Analysis:
        - Motion magnitude: {motion_magnitude:.3f}
        - Direction indicator: {direction:.3f}
        - Embedding dimensions: {len(start_emb)}
        
        Interpretation:
        {"Fast motion detected" if motion_magnitude > 0.5 else "Slower motion"}
        {"Downward motion" if direction < 0 else "Upward/forward motion"}
        """
        return analysis
    
    def generate_code(self, start_emb: np.ndarray, goal_emb: np.ndarray, 
                     feedback: str = "") -> str:
        """
        Generate Blender code from V-JEPA embeddings
        
        Args:
            start_emb: V-JEPA embedding of start video
            goal_emb: V-JEPA embedding of goal video
            feedback: Feedback from physics verifier (if iterating)
        
        Returns:
            Blender Python code string
        """
        analysis = self._analyze_embeddings(start_emb, goal_emb)
        few_shot = self._few_shot_examples()
        
        prompt = f"""
You are a Blender physics expert. Given V-JEPA temporal embeddings representing 
motion patterns between a start and goal animation, generate Blender Python code 
that creates the transformation.

{few_shot}

Current input:
{analysis}

{("Previous attempt feedback: " + feedback) if feedback else ""}

Generate complete Blender Python code following the pattern of examples above.
The code should:
1. Set up physics world with gravity: bpy.ops.rigidbody.world_add() and scene.gravity = (0, 0, -9.81)
2. Create a track with appropriate geometry (e.g., plane rotated to create ramp)
3. Add RIGID BODY physics to track: bpy.ops.rigidbody.object_add(type='PASSIVE') then track.rigid_body.friction = value
   CRITICAL: Use rigid_body.friction, NOT modifiers["Collision"].friction (that's for soft body, not rigid body!)
4. Create a ball with correct physics properties: bpy.ops.rigidbody.object_add(type='ACTIVE') then ball.rigid_body.mass, friction, restitution
5. Set up camera to view the scene properly (location around (5-8, -10 to -12, 4-6), rotation around X axis 60-70 degrees, pointing at scene center)
6. Add lighting (SUN light at location like (5, -5, 10) with energy 6.0)
7. Use parameters inferred from the embedding analysis

CRITICAL API NOTES:
- For passive objects (track): bpy.ops.rigidbody.object_add(type='PASSIVE') then object.rigid_body.friction
- For active objects (ball): bpy.ops.rigidbody.object_add(type='ACTIVE') then object.rigid_body.mass, friction, restitution
- DO NOT use modifiers["Collision"] for rigid body physics - that's wrong!
- Always include camera setup: scene.camera = cam_object

Return ONLY the Python code, no explanation.
"""
        
        if self.client:
            try:
                if self.provider == "gemini":
                    # Gemini API
                    full_prompt = f"You are a Blender physics expert.\n\n{prompt}"
                    response = self.client.generate_content(
                        full_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,
                            top_p=0.95,
                            top_k=40,
                        )
                    )
                    code = response.text
                elif self.provider == "openai":
                    # OpenAI API
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a Blender physics expert."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    code = response.choices[0].message.content
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                
                # Extract code block if wrapped in ```
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0]
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0]
                code = code.strip()
                
                # Auto-fix common API mistakes
                code = self._fix_common_errors(code)
                
                return code
            except Exception as e:
                print(f"❌ LLM error: {e}")
                import traceback
                traceback.print_exc()
                return self._fallback_code()
        else:
            # Fallback: return template code
            return self._fallback_code()
    
    def _fix_common_errors(self, code: str) -> str:
        """Fix common Blender API errors in generated code"""
        import re
        
        original_code = code
        
        # Fix 1: modifiers["Collision"].friction -> rigid_body.friction
        # Pattern: track.modifiers["Collision"].friction
        code = re.sub(r'(\w+)\.modifiers\[["\']Collision["\']\]\.friction', 
                     r'\1.rigid_body.friction', code)
        code = re.sub(r'(\w+)\.modifiers\[["\']Collision["\']\]\.restitution', 
                     r'\1.rigid_body.restitution', code)
        
        # Fix 2: .collision.friction -> .rigid_body.friction
        code = re.sub(r'\.collision\.friction', '.rigid_body.friction', code)
        code = re.sub(r'\.collision\.restitution', '.rigid_body.restitution', code)
        
        # Fix 3: If we see modifiers.new(name="Collision" but no rigidbody.object_add, add it
        if 'modifiers.new' in code and 'COLLISION' in code.upper():
            # Find lines with modifiers.new for collision
            lines = code.split('\n')
            fixed_lines = []
            for i, line in enumerate(lines):
                if 'modifiers.new' in line and ('Collision' in line or 'COLLISION' in line):
                    # Check if rigidbody was already added for this object
                    obj_name = None
                    # Try to find object name from previous lines
                    for j in range(max(0, i-5), i):
                        match = re.search(r'(\w+)\s*=\s*bpy\.context\.object', lines[j])
                        if match:
                            obj_name = match.group(1)
                            break
                    
                    # If we found object and it needs rigid body, replace modifiers.new line
                    if obj_name and f'{obj_name}.rigid_body' not in '\n'.join(lines[:i]):
                        indent = len(line) - len(line.lstrip())
                        # Add rigid body instead
                        fixed_lines.append(' ' * indent + f'bpy.ops.rigidbody.object_add(type=\'PASSIVE\')')
                        continue  # Skip the modifiers.new line
                
                fixed_lines.append(line)
            code = '\n'.join(fixed_lines)
        
        if code != original_code:
            print("⚠️  Auto-fixed collision modifier API → rigid_body API")
        
        return code
    
    def _fallback_code(self) -> str:
        """Fallback code if LLM unavailable"""
        return """
import bpy
import math

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

scene = bpy.context.scene
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()

scene.gravity = (0.0, 0.0, -9.81)
scene.frame_start = 1
scene.frame_end = 300

# Create track
bpy.ops.mesh.primitive_plane_add(size=10, enter_editmode=False, align='WORLD', location=(0, 0, 0))
track = bpy.context.object
track.rotation_euler = (math.radians(25), 0, 0)
bpy.ops.rigidbody.object_add(type='PASSIVE')
track.rigid_body.friction = 0.3

# Create ball
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(0, 2, 1))
ball = bpy.context.object
bpy.ops.rigidbody.object_add(type='ACTIVE')
ball.rigid_body.mass = 1.5
ball.rigid_body.friction = 0.3
ball.rigid_body.restitution = 0.1

# Camera setup - CRITICAL for rendering
bpy.ops.object.camera_add(location=(7.0, -12.0, 6.0))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(70), 0, 0)
scene.camera = cam

# Lighting
bpy.ops.object.light_add(type='SUN', location=(8, -8, 20))
sun = bpy.context.active_object
sun.data.energy = 6.0
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
"""


class PhysicsVerifier:
    """Validates generated code using Blender physics simulation"""
    
    def __init__(self, blender_path: str = "/Applications/Blender.app/Contents/MacOS/Blender"):
        self.blender_path = blender_path
    
    def verify_code(self, code: str, start_video: np.ndarray, 
                   goal_video: np.ndarray) -> Tuple[bool, Dict[str, float], str]:
        """
        Verify generated code by executing in Blender and comparing results
        
        Returns:
            is_valid: Whether code passes all checks
            scores: Per-Frame Fidelity, TFA, PVS
            feedback: Feedback string for refinement
        """
        # Save code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_code_path = f.name
        
        try:
            # Execute code in Blender and render
            result_video = self._execute_and_render(temp_code_path)
            
            # Evaluate
            scores = self._evaluate(result_video, goal_video)
            
            # Check validity
            is_valid = (
                scores['per_frame_fidelity'] > 0.85 and
                scores['temporal_flow_alignment'] > 0.80 and
                scores['physics_validity_score'] > 0.90
            )
            
            # Generate feedback
            feedback = self._generate_feedback(scores, is_valid)
            
            return is_valid, scores, feedback
            
        except Exception as e:
            return False, {}, f"Code execution error: {str(e)}"
        finally:
            Path(temp_code_path).unlink()
    
    def _execute_and_render(self, code_path: str, num_frames: int = 30) -> np.ndarray:
        """Execute Blender code and render result video"""
        import subprocess
        import shutil
        
        # Create wrapper script that executes code and renders frames
        temp_dir = tempfile.mkdtemp()
        wrapper_path = os.path.join(temp_dir, 'render_wrapper.py')
        output_dir = os.path.join(temp_dir, 'rendered_frames')
        os.makedirs(output_dir, exist_ok=True)
        
        wrapper_code = f"""
import bpy
import sys
import os

# Clear existing scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Execute the generated code
with open(r'{code_path}', 'r') as f:
    exec(f.read())

# Ensure camera is set up (if code didn't create one)
scene = bpy.context.scene
if scene.camera is None:
    import math
    # Create camera if missing
    bpy.ops.object.camera_add(location=(7.0, -12.0, 6.0))
    cam = bpy.context.active_object
    cam.rotation_euler = (math.radians(70), 0, 0)
    scene.camera = cam

# Ensure lighting exists
if len([obj for obj in scene.objects if obj.type == 'LIGHT']) == 0:
    import math
    bpy.ops.object.light_add(type='SUN', location=(8, -8, 20))
    sun = bpy.context.active_object
    sun.data.energy = 6.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))

# Setup rendering
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 300
scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.resolution_percentage = 100
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.image_settings.file_format = 'PNG'

# Render sample frames
output_dir = r'{output_dir}'
frame_indices = list(range(1, min(301, scene.frame_end + 1), max(1, (scene.frame_end - 1) // {num_frames - 1})))
if scene.frame_end not in frame_indices:
    frame_indices.append(scene.frame_end)

for i, frame_num in enumerate(frame_indices[:{num_frames}]):
    scene.frame_set(frame_num)
    scene.render.filepath = os.path.join(output_dir, f'frame_{{i:04d}}.png')
    bpy.ops.render.render(write_still=True)

print(f"Rendered {{len(frame_indices)}} frames to {{output_dir}}")
"""
        
        # Write wrapper
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_code)
        
        try:
            # Execute in Blender
            result = subprocess.run(
                [self.blender_path, '--background', '--python', wrapper_path],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode != 0:
                print(f"⚠️  Blender execution failed: {result.stderr[:200]}")
                return np.zeros((num_frames, 224, 224, 3))
            
            # Load rendered frames
            video_frames = []
            frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
            
            for frame_file in frame_files[:num_frames]:
                frame_path = os.path.join(output_dir, frame_file)
                try:
                    if CV2_AVAILABLE:
                        img = cv2.imread(frame_path)
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            video_frames.append(img_rgb)
                    else:
                        from PIL import Image
                        img = Image.open(frame_path)
                        video_frames.append(np.array(img))
                except Exception as e:
                    print(f"⚠️  Failed to load frame {frame_file}: {e}")
                    continue
            
            if video_frames:
                return np.array(video_frames)
            else:
                print("⚠️  No frames rendered")
                return np.zeros((num_frames, 224, 224, 3))
                
        except subprocess.TimeoutExpired:
            print("⚠️  Blender execution timed out")
            return np.zeros((num_frames, 224, 224, 3))
        except Exception as e:
            print(f"⚠️  Error executing Blender code: {e}")
            return np.zeros((num_frames, 224, 224, 3))
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _evaluate(self, result_video: np.ndarray, goal_video: np.ndarray) -> Dict[str, float]:
        """Evaluate result against goal video"""
        scores = {}
        
        # 1. Per-Frame Fidelity (visual similarity)
        if result_video.size > 0 and goal_video.size > 0:
            # Resize if dimensions don't match
            if result_video.shape != goal_video.shape:
                min_frames = min(len(result_video), len(goal_video))
                result_video = result_video[:min_frames]
                goal_video = goal_video[:min_frames]
                
                if result_video.shape[1:3] != goal_video.shape[1:3]:
                    # Resize to match
                    if CV2_AVAILABLE:
                        resized = []
                        for frame in result_video:
                            resized.append(cv2.resize(frame, 
                                (goal_video.shape[2], goal_video.shape[1])))
                        result_video = np.array(resized)
            
            # Compute pixel-level similarity
            if result_video.shape == goal_video.shape:
                mse = np.mean((result_video.astype(float) - goal_video.astype(float)) ** 2)
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / (np.sqrt(mse) + 1e-10))
                # Convert PSNR to similarity score (0-1)
                scores['per_frame_fidelity'] = min(1.0, psnr / 40.0)  # Normalize
            else:
                scores['per_frame_fidelity'] = 0.5  # Fallback
        else:
            scores['per_frame_fidelity'] = 0.0
        
        # 2. Temporal Flow Alignment (motion consistency)
        if len(result_video) > 1 and len(goal_video) > 1:
            try:
                if CV2_AVAILABLE:
                    # Compute optical flow for both videos
                    def compute_flow_consistency(video):
                        flows = []
                        prev = cv2.cvtColor(video[0], cv2.COLOR_RGB2GRAY)
                        for frame in video[1:]:
                            curr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                            flow = cv2.calcOpticalFlowFarneback(
                                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
                            )
                            flows.append(flow)
                            prev = curr
                        return flows
                    
                    result_flows = compute_flow_consistency(result_video)
                    goal_flows = compute_flow_consistency(goal_video)
                    
                    # Compare flow consistency
                    if len(result_flows) == len(goal_flows):
                        flow_diffs = []
                        for rf, gf in zip(result_flows, goal_flows):
                            diff = np.mean(np.abs(rf - gf))
                            flow_diffs.append(diff)
                        
                        # Convert to similarity (lower diff = higher similarity)
                        avg_diff = np.mean(flow_diffs)
                        max_expected_diff = 50.0  # Threshold
                        scores['temporal_flow_alignment'] = max(0.0, 1.0 - avg_diff / max_expected_diff)
                    else:
                        scores['temporal_flow_alignment'] = 0.5
                else:
                    # Fallback: simple frame difference consistency
                    result_diffs = np.diff(result_video, axis=0)
                    goal_diffs = np.diff(goal_video, axis=0)
                    if result_diffs.shape == goal_diffs.shape:
                        diff_sim = 1.0 - np.mean(np.abs(result_diffs - goal_diffs)) / 255.0
                        scores['temporal_flow_alignment'] = max(0.0, diff_sim)
                    else:
                        scores['temporal_flow_alignment'] = 0.5
            except Exception as e:
                print(f"⚠️  TFA computation failed: {e}")
                scores['temporal_flow_alignment'] = 0.5
        else:
            scores['temporal_flow_alignment'] = 0.0
        
        # 3. Physics Validity Score
        # Check for common physics violations in rendered video
        physics_score = 1.0
        
        if result_video.size > 0:
            # Check 1: Objects shouldn't be completely static (if goal has motion)
            if len(goal_video) > 1:
                goal_motion = np.mean(np.abs(np.diff(goal_video, axis=0)))
                result_motion = np.mean(np.abs(np.diff(result_video, axis=0)))
                
                # If goal has motion but result doesn't, penalize
                if goal_motion > 10 and result_motion < 5:
                    physics_score -= 0.3
            
            # Check 2: No sudden jumps (teleportation)
            if len(result_video) > 2:
                frame_diffs = np.diff(result_video, axis=0)
                large_jumps = np.sum(np.abs(frame_diffs) > 100)  # Large pixel changes
                if large_jumps > len(frame_diffs) * 0.3:  # More than 30% are jumps
                    physics_score -= 0.2
            
            # Check 3: Reasonable brightness (not completely black/white)
            avg_brightness = np.mean(result_video)
            if avg_brightness < 10 or avg_brightness > 245:
                physics_score -= 0.1
        
        scores['physics_validity_score'] = max(0.0, min(1.0, physics_score))
        
        return scores
    
    def _generate_feedback(self, scores: Dict[str, float], is_valid: bool) -> str:
        """Generate actionable feedback for code refinement"""
        if is_valid:
            return "✅ Code passes all validation checks!"
        
        feedback_parts = []
        
        if scores.get('per_frame_fidelity', 0) < 0.85:
            feedback_parts.append("Per-frame visual similarity is low. Check geometry and materials match goal.")
        
        if scores.get('temporal_flow_alignment', 0) < 0.80:
            feedback_parts.append("Temporal flow is inconsistent. Motion may be too fast/slow or jerky. Adjust friction and damping.")
        
        if scores.get('physics_validity_score', 0) < 0.90:
            feedback_parts.append("Physics violations detected. Check for: objects floating, interpenetration, unrealistic motion. Adjust mass, friction, or restitution.")
        
        return " | ".join(feedback_parts) if feedback_parts else "Minor improvements needed."


class ZeroShotWorldCoder:
    """Main pipeline: Zero-shot V-JEPA + LLM + Physics Verifier"""
    
    def __init__(self, vjepa_model_path: Optional[str] = None,
                 llm_api_key: Optional[str] = None,
                 llm_provider: str = "gemini",
                 llm_model: str = "gemini-2.0-flash-exp",
                 max_iterations: int = 3):
        self.vjepa = VJEPAEncoder(vjepa_model_path)
        self.llm = LLMCodeGenerator(llm_api_key, model=llm_model, provider=llm_provider)
        self.verifier = PhysicsVerifier()
        self.max_iterations = max_iterations
    
    def transform(self, start_video: np.ndarray, goal_video: np.ndarray,
                 start_video_path: Optional[str] = None,
                 goal_video_path: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
        """
        Transform start video to goal video using zero-shot approach
        
        Args:
            start_video: Start animation frames (T, H, W, C)
            goal_video: Goal animation frames (T, H, W, C)
            start_video_path: Optional path to start video file
            goal_video_path: Optional path to goal video file
        
        Returns:
            final_code: Best Blender code found
            final_scores: Evaluation scores
        """
        print("\n" + "="*70)
        print("Zero-Shot WorldCoder: V-JEPA + LLM + Physics Verifier")
        print("="*70)
        
        # Load videos if paths provided
        if start_video_path:
            start_video = self._load_video(start_video_path)
        if goal_video_path:
            goal_video = self._load_video(goal_video_path)
        
        # Step 1: Extract embeddings with frozen V-JEPA
        print("\n[Step 1] Extracting temporal embeddings with V-JEPA...")
        start_emb = self.vjepa.encode(start_video)
        goal_emb = self.vjepa.encode(goal_video)
        if start_emb is not None and goal_emb is not None:
            print(f"✅ Embeddings extracted: start={start_emb.shape}, goal={goal_emb.shape}")
        else:
            print(f"⚠️  Using placeholder embeddings")
        
        # Step 2-3: Iterative code generation and verification
        feedback = ""
        best_code = None
        best_scores = {}
        
        for iteration in range(self.max_iterations):
            print(f"\n[Iteration {iteration + 1}/{self.max_iterations}]")
            
            # Generate code
            print("  Generating Blender code with LLM...")
            code = self.llm.generate_code(start_emb, goal_emb, feedback)
            print(f"  ✅ Code generated ({len(code)} chars)")
            
            # Verify with physics engine
            print("  Verifying code with physics engine...")
            is_valid, scores, new_feedback = self.verifier.verify_code(
                code, start_video, goal_video
            )
            
            print(f"  Per-Frame Fidelity: {scores.get('per_frame_fidelity', 0):.3f}")
            print(f"  TFA: {scores.get('temporal_flow_alignment', 0):.3f}")
            print(f"  PVS: {scores.get('physics_validity_score', 0):.3f}")
            
            # Track best
            if not best_scores or sum(scores.values()) > sum(best_scores.values()):
                best_code = code
                best_scores = scores
            
            # Check if valid
            if is_valid:
                print("\n✅ Code passes all validation checks!")
                break
            
            # Prepare feedback for next iteration
            feedback = new_feedback
            print(f"  Feedback: {feedback}")
        
        print("\n" + "="*70)
        print(f"Final code length: {len(best_code)} chars")
        print(f"Final scores: {best_scores}")
        print("="*70 + "\n")
        
        return best_code, best_scores
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """Load video from file"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                return np.array(frames)
        except:
            pass
        # Fallback: placeholder
        return np.zeros((30, 224, 224, 3))


# Example usage
if __name__ == "__main__":
    # Initialize
    coder = ZeroShotWorldCoder(
        # vjepa_model_path="/path/to/vjepa/model",
        # llm_api_key="your-api-key",
        max_iterations=3
    )
    
    # Example: Transform animation
    start_video = np.zeros((30, 224, 224, 3))  # Placeholder
    goal_video = np.zeros((30, 224, 224, 3))   # Placeholder
    
    code, scores = coder.transform(start_video, goal_video)
    
    print("\nGenerated code:")
    print(code[:500] + "..." if len(code) > 500 else code)

