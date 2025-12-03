
# Local Autonomous AI Agent (Mac Optimized)

Fully offline AI agent for Apple Silicon (M1/M2/M3/M4). Generates content, AI images, offline narration, and compiles videos.

## Features
- Local LLM text generation (LLaMA 7B)
- Stable Diffusion image generation (512x512)
- Offline TTS via pyttsx3
- Video compilation using moviepy
- Memory stored in JSON

## Folder Structure
```
AI_agent_2/
├─ main.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ memory.json
├─ images/        # empty folder
├─ videos/        # empty folder
```

## Installation
```bash
git clone <YOUR_REPO_URL>
cd AI_agent_2
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

Generated videos are saved in `videos/` and images in `images/`.

---

## Git Setup
```bash
cd AI_agent_2
git init
git add .
git commit -m "Initial commit – M4 Mac Mini AI agent"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/AI_agent_2.git
git push -u origin main
```

After this, your project will be fully on GitHub and you can clone it on any machine.
