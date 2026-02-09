"""
Streamlit Interface for Zero-Shot WorldCoder
Interactive demo for testing the pipeline
"""

import streamlit as st

# Page config MUST be first Streamlit command
st.set_page_config(
    page_title="Zero-Shot WorldCoder",
    page_icon="🎬",
    layout="wide"
)

import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.zero_shot_worldcoder import ZeroShotWorldCoder

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not installed. Video loading may be limited. Install with: pip install opencv-python")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = None
if 'start_video' not in st.session_state:
    st.session_state.start_video = None
if 'goal_video' not in st.session_state:
    st.session_state.goal_video = None

# Title
st.title("🎬 Zero-Shot WorldCoder")
st.markdown("**4D Scene Editing via Physics-Aware Code Generation**")
st.markdown("Transform start animations into goal animations using V-JEPA + LLM + Physics Verifier")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Key input
gemini_key = st.sidebar.text_input(
    "Gemini API Key",
    value="XXXX",
    type="default"
)

# Model paths
vjepa_path = st.sidebar.text_input(
    "V-JEPA Model Path",
    value=str(PROJECT_ROOT / "models" / "vjepa" / "vitl16.pth.tar")
)

max_iterations = st.sidebar.slider("Max Iterations", 1, 5, 3)

# Mode selection
mode = st.sidebar.radio(
    "Input Mode",
    ["Upload Videos", "Use Blender Files", "Demo with Random"]
)

# Main content area
col1, col2 = st.columns(2)

def load_video_from_file(uploaded_file):
    """Load video from uploaded file"""
    if uploaded_file is None:
        return None
    
    if not CV2_AVAILABLE:
        st.error("OpenCV required for video loading. Install with: pip install opencv-python")
        return None
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Load with OpenCV
    cap = cv2.VideoCapture(tmp_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    os.unlink(tmp_path)
    
    if frames:
        return np.array(frames)
    return None

def render_blend_file(blend_path: str, num_frames: int = 30) -> np.ndarray:
    """Render Blender file to numpy array"""
    if not os.path.exists(blend_path):
        return None
    
    temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(temp_dir, 'render_script.py')
    
    render_script = f"""
import bpy
import sys
import os

bpy.ops.wm.open_mainfile(filepath=r'{blend_path}')
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 300
scene.render.resolution_x = 224
scene.render.resolution_y = 224
scene.render.resolution_percentage = 100
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.image_settings.file_format = 'PNG'

output_dir = r'{temp_dir}'
os.makedirs(output_dir, exist_ok=True)

frame_indices = list(range(1, min(301, scene.frame_end + 1), max(1, (scene.frame_end - 1) // {num_frames - 1})))
if scene.frame_end not in frame_indices:
    frame_indices.append(scene.frame_end)

for i, frame_num in enumerate(frame_indices[:{num_frames}]):
    scene.frame_set(frame_num)
    frame_path = os.path.join(output_dir, f'frame_{{i:04d}}.png')
    scene.render.filepath = frame_path
    try:
        bpy.ops.render.render(write_still=True)
    except:
        pass
"""
    
    with open(script_path, 'w') as f:
        f.write(render_script)
    
    blender_path = '/Applications/Blender.app/Contents/MacOS/Blender'
    subprocess.run(
        [blender_path, '--background', '--python', script_path],
        capture_output=True,
        timeout=120
    )
    
    # Load frames
    video_frames = []
    if CV2_AVAILABLE:
        for i in range(num_frames):
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img_rgb.shape[2] == 4:
                        img_rgb = img_rgb[:, :, :3]
                    video_frames.append(img_rgb)
    else:
        # Fallback to PIL
        try:
            from PIL import Image
            for i in range(num_frames):
                frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                if os.path.exists(frame_path):
                    img = Image.open(frame_path)
                    img_array = np.array(img)
                    if img_array.shape[2] == 4:
                        img_array = img_array[:, :, :3]
                    video_frames.append(img_array)
        except ImportError:
            pass
    
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    if video_frames:
        return np.array(video_frames)
    return None

# Input section
if mode == "Upload Videos":
    st.header("📤 Upload Videos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Start Video")
        start_file = st.file_uploader("Upload start video", type=['mp4', 'avi', 'mov'], key='start')
        if start_file:
            with st.spinner("Loading start video..."):
                start_video = load_video_from_file(start_file)
                if start_video is not None:
                    st.session_state.start_video = start_video
                    st.success(f"Loaded: {start_video.shape}")
                    st.video(start_file)
        
        # Show previously loaded video if exists
        if st.session_state.start_video is not None and start_file is None:
            st.info(f"✓ Start video loaded: {st.session_state.start_video.shape}")
            # Show first frame as preview
            st.image(st.session_state.start_video[0], caption="Start Video (Frame 0)")
    
    with col2:
        st.subheader("Goal Video")
        goal_file = st.file_uploader("Upload goal video", type=['mp4', 'avi', 'mov'], key='goal')
        if goal_file:
            with st.spinner("Loading goal video..."):
                goal_video = load_video_from_file(goal_file)
                if goal_video is not None:
                    st.session_state.goal_video = goal_video
                    st.success(f"Loaded: {goal_video.shape}")
                    st.video(goal_file)
        
        # Show previously loaded video if exists
        if st.session_state.goal_video is not None and goal_file is None:
            st.info(f"✓ Goal video loaded: {st.session_state.goal_video.shape}")
            # Show first frame as preview
            st.image(st.session_state.goal_video[0], caption="Goal Video (Frame 0)")

elif mode == "Use Blender Files":
    st.header("🎨 Select Blender Files")
    
    dataset_dir = PROJECT_ROOT / "dataset" / "blender_files"
    
    if dataset_dir.exists():
        blend_files = sorted(list(dataset_dir.glob("*.blend")))
        start_files = [f for f in blend_files if "start" in f.name]
        goal_files = [f for f in blend_files if "goal" in f.name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Start Scene")
            selected_start = st.selectbox(
                "Choose start .blend file",
                start_files,
                format_func=lambda x: x.name
            )
            if selected_start and st.button("Load Start Scene", key='load_start'):
                with st.spinner("Rendering start scene..."):
                    start_video = render_blend_file(str(selected_start), num_frames=30)
                    if start_video is not None:
                        st.session_state.start_video = start_video
                        st.success(f"Rendered: {start_video.shape}")
                        # Show sample frames
                        st.image(start_video[0], caption="Frame 0")
            
            # Always show previously loaded video if exists
            if st.session_state.start_video is not None:
                if selected_start:  # Only show info if we have a file selected
                    st.info(f"✓ Start scene loaded: {st.session_state.start_video.shape}")
                if st.session_state.start_video.shape[0] > 0:
                    st.image(st.session_state.start_video[0], caption="Start Scene (Frame 0)")
        
        with col2:
            st.subheader("Goal Scene")
            selected_goal = st.selectbox(
                "Choose goal .blend file",
                goal_files,
                format_func=lambda x: x.name
            )
            if selected_goal and st.button("Load Goal Scene", key='load_goal'):
                with st.spinner("Rendering goal scene..."):
                    goal_video = render_blend_file(str(selected_goal), num_frames=30)
                    if goal_video is not None:
                        st.session_state.goal_video = goal_video
                        st.success(f"Rendered: {goal_video.shape}")
                        st.image(goal_video[0], caption="Frame 0")
            
            # Always show previously loaded video if exists
            if st.session_state.goal_video is not None:
                if selected_goal:  # Only show info if we have a file selected
                    st.info(f"✓ Goal scene loaded: {st.session_state.goal_video.shape}")
                if st.session_state.goal_video.shape[0] > 0:
                    st.image(st.session_state.goal_video[0], caption="Goal Scene (Frame 0)")
    else:
        st.error(f"Dataset directory not found: {dataset_dir}")

else:  # Demo mode
    st.header("🎲 Demo Mode")
    st.info("Using random placeholder videos for demonstration")
    
    if st.button("Generate Demo Videos"):
        start_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 150 + 100
        goal_video = np.random.rand(30, 224, 224, 3).astype(np.uint8) * 150 + 100
        st.session_state.start_video = start_video
        st.session_state.goal_video = goal_video
        st.success("Demo videos generated!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(start_video[0], caption="Start Video (Frame 0)")
        with col2:
            st.image(goal_video[0], caption="Goal Video (Frame 0)")

# Run pipeline
st.header("🚀 Run Pipeline")

if st.button("✨ Transform Start → Goal", type="primary"):
    # Check if we have videos in session state
    if st.session_state.start_video is None or st.session_state.goal_video is None:
        st.error("Please load both start and goal videos/scenes first!")
    else:
        start_video = st.session_state.start_video
        goal_video = st.session_state.goal_video
        # Initialize pipeline
        with st.spinner("Initializing pipeline..."):
            try:
                pipeline = ZeroShotWorldCoder(
                    vjepa_model_path=vjepa_path if os.path.exists(vjepa_path) else None,
                    llm_api_key=gemini_key,
                    llm_provider='gemini',
                    llm_model='gemini-2.0-flash-exp',
                    max_iterations=max_iterations
                )
                st.session_state.pipeline = pipeline
                st.success("Pipeline initialized!")
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.stop()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run transformation
        try:
            status_text.text("Step 1/3: Extracting V-JEPA embeddings...")
            progress_bar.progress(20)
            
            status_text.text("Step 2/3: Generating code with Gemini...")
            progress_bar.progress(50)
            
            status_text.text("Step 3/3: Verifying physics...")
            progress_bar.progress(80)
            
            code, scores = pipeline.transform(st.session_state.start_video, st.session_state.goal_video)
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            st.session_state.generated_code = code
            st.session_state.results = scores
            
        except Exception as e:
            st.error(f"Transformation failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# Results section
if st.session_state.generated_code:
    st.header("📊 Results")
    
    # Scores
    if st.session_state.results:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fidelity = st.session_state.results.get('per_frame_fidelity', 0)
            st.metric("Per-Frame Fidelity", f"{fidelity:.3f}")
        
        with col2:
            tfa = st.session_state.results.get('temporal_flow_alignment', 0)
            st.metric("Temporal Flow Alignment", f"{tfa:.3f}")
        
        with col3:
            pvs = st.session_state.results.get('physics_validity_score', 0)
            st.metric("Physics Validity Score", f"{pvs:.3f}")
    
    # Generated code
    st.subheader("📝 Generated Blender Code")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.code(st.session_state.generated_code, language='python')
    
    with col2:
        st.download_button(
            label="📥 Download Code",
            data=st.session_state.generated_code,
            file_name="generated_blender_code.py",
            mime="text/x-python"
        )
        
        code_stats = {
            "Lines": len(st.session_state.generated_code.split('\n')),
            "Characters": len(st.session_state.generated_code),
            "Has Physics": 'rigidbody' in st.session_state.generated_code.lower(),
            "Has Objects": 'sphere' in st.session_state.generated_code.lower() or 'cube' in st.session_state.generated_code.lower()
        }
        
        st.json(code_stats)
    
    # Instructions
    st.subheader("💡 How to Use Generated Code")
    st.markdown("""
    1. Open Blender
    2. Copy the generated code
    3. Open Blender's Scripting workspace
    4. Paste and run the code
    5. Play the animation to see the result!
    """)

# Footer
st.markdown("---")
st.markdown("**Zero-Shot WorldCoder** | V-JEPA + Gemini + Physics Verifier | Zero-shot 4D Scene Editing")

