from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from generator import load_csm_1b
import torchaudio
import os

# ensure the runtime uses the baked-in HF cache (/opt/hf_cache)
os.environ.setdefault("HF_HOME", "/opt/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ.get("HF_HOME"))
os.environ.setdefault("HF_DATASETS_CACHE", os.environ.get("HF_HOME"))
# make sure if runtime needs a token (some HF code reads env) it's present
if "HUGGINGFACE_TOKEN" in os.environ:
    os.environ.setdefault("HUGGINGFACE_TOKEN", os.environ["HUGGINGFACE_TOKEN"])

app = FastAPI(title="CSM Text-to-Speech API")

# Determine device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load the CSM model
generator = load_csm_1b(device=device)

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

@app.post("/generate")
async def generate_speech(request: TTSRequest):
    try:
        # Generate audio
        audio = generator.generate(
            text=request.text,
            speaker=request.speaker,
            context=[],
            max_audio_length_ms=10_000,
        )

        # Save to temporary file
        temp_file = f"/tmp/audio_{torch.randint(0, 1000000, (1,)).item()}.wav"
        torchaudio.save(temp_file, audio.unsqueeze(0).cpu(), generator.sample_rate)

        # Read the file and return as bytes
        with open(temp_file, "rb") as f:
            audio_bytes = f.read()

        # Clean up
        os.remove(temp_file)

        return {
            "audio": audio_bytes,
            "sample_rate": generator.sample_rate,
            "text": request.text,
            "speaker": request.speaker
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
