# Streamlit Interface Guide

## Quick Start

1. **Run the app:**
   ```bash
   ./run_streamlit.sh
   ```
   Or directly:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open in browser:**
   - Automatically opens at `http://localhost:8501`
   - Or manually navigate to that URL

## Features

### 📤 Three Input Modes

1. **Upload Videos**: Upload MP4/AVI/MOV files
   - Upload start video
   - Upload goal video
   - Run transformation

2. **Use Blender Files**: Load from your dataset
   - Select start `.blend` file
   - Select goal `.blend` file
   - Renders automatically

3. **Demo with Random**: Quick test with placeholder videos
   - No files needed
   - Instant demo

### ⚙️ Configuration

- **Gemini API Key**: Set your API key
- **V-JEPA Model Path**: Path to model checkpoint
- **Max Iterations**: Number of refinement iterations (1-5)

### 📊 Results Display

- **Metrics Dashboard**: 
  - Per-Frame Fidelity
  - Temporal Flow Alignment
  - Physics Validity Score

- **Generated Code**:
  - Full Python code display
  - Download button
  - Code statistics

## Usage Tips

1. **First Time Setup**:
   - Ensure V-JEPA model path is correct
   - Enter your Gemini API key
   - Start with "Demo with Random" to test

2. **Blender Files**:
   - Files should be in `/dataset/blender_files/`
   - Naming: `start_XXXX.blend` and `goal_XXXX.blend`

3. **Video Upload**:
   - Supports MP4, AVI, MOV
   - Videos will be automatically processed
   - First few frames displayed

4. **Running Transformations**:
   - Click "✨ Transform Start → Goal"
   - Progress bar shows status
   - Results appear automatically

5. **Using Generated Code**:
   - Download the Python file
   - Open Blender → Scripting workspace
   - Paste and run
   - Watch the animation!

## Troubleshooting

- **"Pipeline failed"**: Check API key and model paths
- **"Videos not loading"**: Ensure video format is supported
- **"Blender files not found"**: Check dataset directory path
- **Slow rendering**: Blender rendering takes time (1-2 min per file)

## Features Coming Soon

- Real-time progress for each pipeline step
- Video preview comparison
- Interactive code editing
- Export rendered result videos





