
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
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with local path if needed
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
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
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
        print(f"[INFO] Generated content:
{content}")

        # Save to memory
        memory = {"topic": topic, "content": content}
        with open("memory.json", "w") as f:
            json.dump(memory, f, indent=4)

        # Generate scene images
        scenes = [s for s in content.split("
") if s.strip()]
        if not scenes:
            scenes = [f"Scene {i+1}" for i in range(3)]
        for i, scene in enumerate(scenes):
            scene_text = scene.strip() or f"Scene {i+1}"
            prompt = f"{scene_text}, cartoon style, vibrant colors"
            filename = f"images/scene_{i}.png"
            generate_image(prompt, filename)

        # Generate offline narration
        audio_file = f"videos/{topic.replace(' ', '_')}_audio.mp3"
        text_to_speech(content, audio_file)

        # Compile video
        audio = AudioFileClip(audio_file)
        per_scene = max(audio.duration / max(len(scenes), 1), 0.1)
        clips = [ImageClip(f"images/scene_{i}.png").set_duration(per_scene) for i in range(len(scenes))]
        video = concatenate_videoclips(clips)
        video = video.set_audio(audio)
        video_file = f"videos/{topic.replace(' ', '_')}_video.mp4"
        video.write_videofile(video_file, fps=24)

        print(f"[INFO] Video saved at: {video_file}")

    # --- Run Agent ---
    if __name__ == "__main__":
        topics = ["Joke Factory", "Daily Creativity Challenge"]
        for topic in topics:
            run_agent(topic)
