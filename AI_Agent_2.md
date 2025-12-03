# AI Local Agent for M4 Mac Mini

Fully offline AI agent optimized for Apple Silicon (M1/M2/M3/M4). This autonomous agent generates creative content, AI images, offline narration, and compiles videos using locally hosted models.

## 🚀 Features

- **Local LLM Text Generation** - Uses LLaMA 7B for content creation
- **Stable Diffusion Image Generation** - Creates 512x512 cartoon-style images
- **Offline Text-to-Speech** - No internet required for narration
- **Video Compilation** - Automatically combines images and audio
- **JSON Memory Storage** - Tracks generated content locally

## 📁 Project Structure

```
ai_local_agent_m4/
├─ main.py              # Main agent script
├─ requirements.txt     # Python dependencies
├─ README.md            # Documentation
├─ .gitignore          # Git ignore rules
├─ memory.json         # Agent memory storage
├─ images/             # Generated scene images
└─ videos/             # Compiled videos and audio
```

## 🛠️ Installation

### Prerequisites
- Mac with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- Git

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/<YOUR_USERNAME>/ai_local_agent_m4.git
cd ai_local_agent_m4
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download models** (First run will auto-download)
```bash
python main.py
```

## 📝 File Contents

### main.py
```python
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import pyttsx3
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# --- Setup device for Apple M1/M2/M3/M4 Macs ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# --- Setup folders ---
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# --- Load local LLM ---
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with local path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def generate_text(prompt, max_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Load Stable Diffusion ---
print("[INFO] Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to(device)

def generate_image(prompt, filename):
    image = pipe(prompt, height=512, width=512).images[0]
    image.save(filename)
    print(f"[INFO] Generated image: {filename}")

# --- Offline TTS ---
engine = pyttsx3.init()
def text_to_speech(text, filename):
    engine.save_to_file(text, filename)
    engine.runAndWait()
    print(f"[INFO] Generated audio: {filename}")

# --- Main Agent Function ---
def run_agent(topic):
    print(f"[INFO] Generating content for topic: {topic}...")
    content = generate_text(f"Generate 3 creative items for '{topic}'")
    print(f"[INFO] Generated content:\n{content}")

    # Save to memory
    memory = {"topic": topic, "content": content}
    with open("memory.json", "w") as f:
        json.dump(memory, f, indent=4)

    # Generate scene images
    scenes = content.split("\n")
    for i, scene in enumerate(scenes):
        scene_text = scene.strip() or f"Scene {i+1}"
        prompt = f"{scene_text}, cartoon style, vibrant colors"
        filename = f"images/scene_{i}.png"
        generate_image(prompt, filename)

    # Generate offline narration
    audio_file = f"videos/{topic}_audio.mp3"
    text_to_speech(content, audio_file)

    # Compile video
    audio = AudioFileClip(audio_file)
    clips = [ImageClip(f"images/scene_{i}.png").set_duration(audio.duration / len(scenes))
             for i in range(len(scenes))]
    video = concatenate_videoclips(clips)
    video = video.set_audio(audio)
    video_file = f"videos/{topic}_video.mp4"
    video.write_videofile(video_file, fps=24)

    print(f"[INFO] Video saved at: {video_file}")

# --- Run Agent ---
if __name__ == "__main__":
    topics = ["Joke Factory", "Daily Creativity Challenge"]
    for topic in topics:
        run_agent(topic)
```

### requirements.txt
```
torch
transformers
sentencepiece
diffusers
accelerate
pyttsx3
moviepy
```

### .gitignore
```
videos/
images/
memory.json
__pycache__/
*.pyc
```

### memory.json
```json
{}
```

## 🎮 Usage

Run the agent with default topics:
```bash
python main.py
```

Customize topics by editing the `topics` list in `main.py`:
```python
topics = ["Your Custom Topic", "Another Topic"]
```

## 📤 Output

- **Images**: Saved in `images/` folder as `scene_0.png`, `scene_1.png`, etc.
- **Audio**: Saved in `videos/` folder as `{topic}_audio.mp3`
- **Video**: Final compiled video saved as `videos/{topic}_video.mp4`
- **Memory**: Content history stored in `memory.json`

## ⚙️ System Requirements

- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 20GB free space for models
- **OS**: macOS 12.0+ with Apple Silicon

## 🔧 Troubleshooting

**MPS not available**: Ensure you're running on Apple Silicon Mac with macOS 12.0+

**Out of memory**: Reduce batch size or use smaller models

**Model download fails**: Check internet connection for first-time setup

## 📄 License

MIT License - Feel free to modify and distribute

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss changes.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built for Apple Silicon | Fully Offline | Privacy-First**
