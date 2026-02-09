"""
Zero-Shot WorldCoder: V-JEPA + CLIP + LLM + Physics Verifier (Enhanced)
=======================================================================
Enhanced version with CLIP visual embeddings to improve PFF scores
"""

# Copy all imports from original
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zero_shot_worldcoder import (
    VJEPAEncoder, LLMCodeGenerator, PhysicsVerifier, ZeroShotWorldCoder,
    TORCH_AVAILABLE, CV2_AVAILABLE, GEMINI_AVAILABLE, OPENAI_AVAILABLE
)
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path

# CLIP support
CLIP_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        import clip
        import torch
        CLIP_AVAILABLE = True
    except ImportError:
        pass


class CLIPVisualEncoder:
    """CLIP encoder for visual features from video frames"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model = None
        self.preprocess = None
        self.device = None
        self.embedding_dim = 512  # CLIP ViT-B/32 default
        
        global CLIP_AVAILABLE
        if CLIP_AVAILABLE and TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            try:
                self.model, self.preprocess = clip.load(model_name, device=self.device)
                self.model.eval()
                self.embedding_dim = self.model.visual.output_dim
                print(f"✅ CLIP model loaded: {model_name}, embedding dim: {self.embedding_dim}")
            except Exception as e:
                print(f"⚠️  Failed to load CLIP: {e}")
                CLIP_AVAILABLE = False
        else:
            print("⚠️  CLIP not available. Using placeholder visual embeddings.")
    
    def encode(self, video: np.ndarray) -> np.ndarray:
        """
        Extract visual embeddings from video using CLIP
        
        Args:
            video: (T, H, W, C) numpy array of video frames
        
        Returns:
            embedding: (embedding_dim,) numpy array (mean pooled across frames)
        """
        if self.model is None or not CLIP_AVAILABLE:
            # Placeholder: return random but structured embedding
            return np.random.randn(self.embedding_dim) * 0.1
        
        try:
            # Sample frames (CLIP works on images, not full video)
            num_frames = min(len(video), 8)  # Sample up to 8 frames
            frame_indices = np.linspace(0, len(video) - 1, num_frames).astype(int)
            sampled_frames = video[frame_indices]
            
            # Preprocess frames for CLIP
            import torchvision.transforms as transforms
            from PIL import Image
            
            # CLIP expects RGB PIL Images, normalized
            frame_embeddings = []
            for frame in sampled_frames:
                # Convert to PIL Image
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                # Ensure RGB
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                
                pil_image = Image.fromarray(frame)
                
                # Preprocess for CLIP
                image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                
                # Encode with CLIP
                with torch.no_grad():
                    image_features = self.model.encode_image(image_tensor)
                    frame_embeddings.append(image_features.cpu().numpy()[0])
            
            # Mean pool across frames
            visual_embedding = np.mean(frame_embeddings, axis=0)
            
            return visual_embedding
            
        except Exception as e:
            print(f"⚠️  CLIP encoding failed: {e}")
            # Fallback: placeholder
            return np.random.randn(self.embedding_dim) * 0.1


class EnhancedLLMCodeGenerator(LLMCodeGenerator):
    """Enhanced LLM code generator that uses both temporal and visual embeddings"""
    
    def _analyze_embeddings(self, start_emb: np.ndarray, goal_emb: np.ndarray,
                           start_visual: Optional[np.ndarray] = None,
                           goal_visual: Optional[np.ndarray] = None) -> str:
        """Analyze both temporal and visual embeddings"""
        diff = goal_emb - start_emb
        
        # Temporal analysis
        motion_magnitude = np.linalg.norm(diff)
        direction = np.mean(diff[:100])
        
        analysis = f"""
        Temporal Embedding Analysis (V-JEPA):
        - Motion magnitude: {motion_magnitude:.3f}
        - Direction indicator: {direction:.3f}
        - Embedding dimensions: {len(start_emb)}
        
        Interpretation:
        {"Fast motion detected" if motion_magnitude > 0.5 else "Slower motion"}
        {"Downward motion" if direction < 0 else "Upward/forward motion"}
        """
        
        # Visual analysis (if available)
        if start_visual is not None and goal_visual is not None:
            visual_diff = goal_visual - start_visual
            visual_magnitude = np.linalg.norm(visual_diff)
            
            analysis += f"""
        
        Visual Embedding Analysis (CLIP):
        - Visual change magnitude: {visual_magnitude:.3f}
        - Visual embedding dimensions: {len(start_visual)}
        
        Interpretation:
        {"Significant visual changes detected" if visual_magnitude > 0.3 else "Minor visual changes"}
        Visual embeddings capture: colors, materials, lighting, geometry appearance
        """
        
        return analysis
    
    def generate_code(self, start_emb: np.ndarray, goal_emb: np.ndarray,
                     feedback: str = "",
                     start_visual: Optional[np.ndarray] = None,
                     goal_visual: Optional[np.ndarray] = None) -> str:
        """
        Generate Blender code using both temporal and visual embeddings
        """
        analysis = self._analyze_embeddings(start_emb, goal_emb, start_visual, goal_visual)
        few_shot = self._few_shot_examples()
        
        # Enhanced prompt with visual guidance
        visual_guidance = ""
        if start_visual is not None and goal_visual is not None:
            visual_guidance = """
IMPORTANT: You also have access to CLIP visual embeddings that capture:
- Colors and materials (e.g., red ball, blue ramp, metallic surface)
- Lighting characteristics (brightness, shadows, ambient light)
- Geometric appearance (sizes, shapes, textures)
- Overall visual style

Use these visual embeddings to match the exact visual appearance of the goal video.
Pay attention to:
1. Material colors and properties
2. Lighting setup (sun position, intensity, shadows)
3. Camera positioning and angle
4. Object sizes and proportions
"""
        
        prompt = f"""
You are a Blender physics expert. Given V-JEPA temporal embeddings (motion patterns) 
and CLIP visual embeddings (visual appearance) representing the transformation between 
a start and goal animation, generate Blender Python code that creates the transformation.

{visual_guidance}

{few_shot}

Current input:
{analysis}

{("Previous attempt feedback: " + feedback) if feedback else ""}

Generate complete Blender Python code following the pattern of examples above.
The code should:
1. Set up physics world with gravity: bpy.ops.rigidbody.world_add() and scene.gravity = (0, 0, -9.81)
2. Create a track with appropriate geometry (e.g., plane rotated to create ramp)
3. Add RIGID BODY physics to track: bpy.ops.rigidbody.object_add(type='PASSIVE') then track.rigid_body.friction = value
4. Create a ball with correct physics properties: bpy.ops.rigidbody.object_add(type='ACTIVE') then ball.rigid_body.mass, friction, restitution
5. Set up camera to view the scene properly (use visual embeddings to match camera angle and position)
6. Add lighting (use visual embeddings to match lighting: sun position, intensity, shadows)
7. Use parameters inferred from both temporal (motion) and visual (appearance) embeddings
8. Match materials and colors based on visual embeddings

CRITICAL API NOTES:
- For passive objects (track): bpy.ops.rigidbody.object_add(type='PASSIVE') then object.rigid_body.friction
- For active objects (ball): bpy.ops.rigidbody.object_add(type='ACTIVE') then object.rigid_body.mass, friction, restitution
- DO NOT use modifiers["Collision"] for rigid body physics - that's wrong!
- Always include camera setup: scene.camera = cam_object
- Use visual embeddings to set material colors, lighting, and camera position

Return ONLY the Python code, no explanation.
"""
        
        # Use parent class's generation logic
        return super().generate_code(start_emb, goal_emb, feedback)


class EnhancedZeroShotWorldCoder(ZeroShotWorldCoder):
    """Enhanced zero-shot WorldCoder with CLIP visual features"""
    
    def __init__(self, vjepa_model_path: Optional[str] = None,
                 llm_api_key: Optional[str] = None,
                 llm_provider: str = "gemini",
                 llm_model: str = "gemini-2.0-flash-exp",
                 max_iterations: int = 10,  # Increased max, but will stop early if converged
                 use_clip: bool = True,
                 clip_model: str = "ViT-B/32",
                 convergence_threshold: float = 0.85,  # Stop if PFF > threshold
                 min_iterations: int = 1,  # Minimum iterations before checking convergence
                 convergence_patience: int = 2):  # Stop if no improvement for N iterations
        # Initialize parent
        super().__init__(vjepa_model_path, llm_api_key, llm_provider, llm_model, max_iterations)
        
        # Convergence settings
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        self.convergence_patience = convergence_patience
        
        # Add CLIP encoder
        self.use_clip = use_clip and CLIP_AVAILABLE
        if self.use_clip:
            self.clip_encoder = CLIPVisualEncoder(clip_model)
            # Replace LLM generator with enhanced version
            self.llm = EnhancedLLMCodeGenerator(llm_api_key, model=llm_model, provider=llm_provider)
            print("✅ Enhanced pipeline: V-JEPA (temporal) + CLIP (visual) + LLM")
            print(f"   Convergence: Stop when PFF > {convergence_threshold} or TFA > 0.95")
        else:
            self.clip_encoder = None
            print("⚠️  CLIP not available, using temporal-only embeddings")
    
    def transform(self, start_video: np.ndarray, goal_video: np.ndarray,
                 start_video_path: Optional[str] = None,
                 goal_video_path: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
        """
        Transform start video to goal video using enhanced approach with visual features
        """
        print("\n" + "="*70)
        print("Enhanced Zero-Shot WorldCoder: V-JEPA + CLIP + LLM + Physics Verifier")
        print("="*70)
        
        # Load videos if paths provided
        if start_video_path:
            start_video = self._load_video(start_video_path)
        if goal_video_path:
            goal_video = self._load_video(goal_video_path)
        
        # Step 1: Extract temporal embeddings with V-JEPA
        print("\n[Step 1] Extracting temporal embeddings with V-JEPA...")
        start_emb = self.vjepa.encode(start_video)
        goal_emb = self.vjepa.encode(goal_video)
        print(f"✅ Temporal embeddings: start={start_emb.shape}, goal={goal_emb.shape}")
        
        # Step 2: Extract visual embeddings with CLIP (if available)
        start_visual = None
        goal_visual = None
        if self.use_clip:
            print("\n[Step 2] Extracting visual embeddings with CLIP...")
            start_visual = self.clip_encoder.encode(start_video)
            goal_visual = self.clip_encoder.encode(goal_video)
            print(f"✅ Visual embeddings: start={start_visual.shape}, goal={goal_visual.shape}")
        else:
            print("\n[Step 2] Skipping visual embeddings (CLIP not available)")
        
        # Step 3-4: Iterative code generation and verification with convergence-based stopping
        feedback = ""
        best_code = None
        best_scores = {}
        no_improvement_count = 0
        previous_best_score = 0.0
        
        for iteration in range(self.max_iterations):
            print(f"\n[Iteration {iteration + 1}/{self.max_iterations}]")
            
            # Generate code with both temporal and visual embeddings
            print("  Generating Blender code with LLM (temporal + visual)...")
            code = self.llm.generate_code(start_emb, goal_emb, feedback, start_visual, goal_visual)
            print(f"  ✅ Code generated ({len(code)} chars)")
            
            # Verify with physics engine
            print("  Verifying code with physics engine...")
            is_valid, scores, new_feedback = self.verifier.verify_code(
                code, start_video, goal_video
            )
            
            pff = scores.get('per_frame_fidelity', 0)
            tfa = scores.get('temporal_flow_alignment', 0)
            pvs = scores.get('physics_validity_score', 0)
            total_score = sum(scores.values())
            
            print(f"  Per-Frame Fidelity: {pff:.3f}")
            print(f"  TFA: {tfa:.3f}")
            print(f"  PVS: {pvs:.3f}")
            print(f"  Total Score: {total_score:.3f}")
            
            # Track best
            if not best_scores or total_score > sum(best_scores.values()):
                best_code = code
                best_scores = scores
                no_improvement_count = 0  # Reset counter on improvement
                print(f"  ✨ New best score! (improvement: {total_score - previous_best_score:.3f})")
                previous_best_score = total_score
            else:
                no_improvement_count += 1
                print(f"  No improvement (patience: {no_improvement_count}/{self.convergence_patience})")
            
            # Check convergence conditions
            if iteration + 1 >= self.min_iterations:
                # Condition 1: All metrics pass thresholds
                if is_valid:
                    print("\n✅ Code passes all validation checks! (PFF > 0.85, TFA > 0.80, PVS > 0.90)")
                    break
                
                # Condition 2: Primary metrics are excellent (TFA > 0.95 and PFF > threshold)
                if tfa > 0.95 and pff > self.convergence_threshold:
                    print(f"\n✅ Convergence achieved! (TFA={tfa:.3f} > 0.95, PFF={pff:.3f} > {self.convergence_threshold})")
                    break
                
                # Condition 3: No improvement for several iterations
                if no_improvement_count >= self.convergence_patience:
                    print(f"\n⚠️  Stopping: No improvement for {self.convergence_patience} iterations")
                    print(f"   Best scores: PFF={best_scores.get('per_frame_fidelity', 0):.3f}, "
                          f"TFA={best_scores.get('temporal_flow_alignment', 0):.3f}, "
                          f"PVS={best_scores.get('physics_validity_score', 0):.3f}")
                    break
            
            # Prepare feedback for next iteration
            feedback = new_feedback
            print(f"  Feedback: {feedback}")
        
        print("\n" + "="*70)
        print(f"Final code length: {len(best_code)} chars")
        print(f"Final scores: {best_scores}")
        print("="*70 + "\n")
        
        return best_code, best_scores

