# WorldCoder: Physically Consistent 4D Scene Editing with Multimodal Foundation Models

## Abstract

WorldCoder introduces a new paradigm for 4D (space–time) scene editing via Blender/Python code, extending static 3D editing benchmarks such as BlenderGym into the temporal and physical domains. While recent vision–language models (VLMs) and multimodal LLMs have shown the ability to generate accurate Blender code for static render transformations, real-world creative and simulation workflows involve dynamic scenes—animations, moving lights, deforming materials, and interacting rigid bodies—where edits must satisfy both temporal continuity and physical realism. WorldCoder defines a structured benchmark and dataset that challenges models to translate a pair of short video clips—representing a start and goal state—into a sequence of Blender/Python edit operations that transform the first animation into the second without breaking temporal coherence or physical laws.

Each benchmark instance includes paired start and goal animations rendered from procedural Blender scenes, their corresponding configuration files, and the ground-truth edit script. Dynamic scenarios include object motion, light trajectory modifications, keyframed transformations, and surface/material evolution under physically simulated constraints such as gravity and collision. All generated trajectories are validated using Blender's physics engine to ensure stability and conservation of energy before inclusion in the dataset. To facilitate scalability, the dataset is procedurally synthesized, allowing automatic generation of thousands of physically valid dynamic scenes with controllable difficulty levels.

Evaluation combines three complementary criteria. First, **Per-Frame Fidelity** measures frame-wise visual and geometric similarity using pixel-level and embedding-based distances (e.g., LPIPS, CLIP cosine). Second, **Temporal Flow Alignment (TFA)** evaluates the consistency of inter-frame optical-flow trajectories between the generated and target videos, penalizing motion discontinuities or jitter. Third, the **Physics Validity Score (PVS)** captures the proportion of frames that remain collision-free, gravity-consistent, and dynamically stable when re-simulated. Together these metrics quantify not only perceptual accuracy but also adherence to physically plausible dynamics—an essential property for 4D understanding.

WorldCoder builds on the generator–verifier paradigm but adds a physics-in-the-loop verifier. The generator (a multimodal foundation model) proposes multiple candidate Blender edit scripts from the start/goal video pair. Each script is rendered into a short animation and evaluated by both visual–temporal scorers and a differentiable physics verifier. The latter enforces rigid-body stability, detects intersection errors, and checks for violations of momentum or gravity. A compute-budgeted selection module then chooses the best script or feeds the verifier feedback back to the generator for refinement, establishing an iterative, self-correcting loop that balances creativity and physical consistency.

## Pipeline

```
Start & Goal Videos → Generator (VLM+LLM → Code) → Code Proposals → 
Render Engine (Blender) → Physics Verifier (Collision, Gravity, Energy) → 
Temporal/Visual Scorer (Fidelity + Flow Align) → Feedback Signal → Generator
```

## Keywords

4D scene editing, Blender, multimodal reasoning, physics-in-the-loop, temporal coherence, foundation models

## Structure

- `generator/`: VLM+LLM code generation modules
- `verifier/`: Physics verifier and temporal/visual scoring
- `renderer/`: Blender integration and rendering pipeline
- `dataset/`: Benchmark dataset and ground-truth scripts
- `evaluation/`: Metrics computation (Fidelity, TFA, PVS)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Setup Blender environment
# ...
```

## Citation

```bibtex
@article{worldcoder2024,
  title={WorldCoder: Physically Consistent 4D Scene Editing with Multimodal Foundation Models},
  author={...},
  year={2024}
}
```
