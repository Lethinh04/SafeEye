from transformers import VitsModel, AutoTokenizer
import torch
import sounddevice as sd
import numpy as np

# Load model và tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

# Nhập câu tiếng Việt
text = input("Nhập câu tiếng Việt: ")

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Inference
with torch.no_grad():
    output = model(**inputs).waveform

# Convert sang numpy
audio = output.squeeze().cpu().numpy()

# Phát âm thanh
sd.play(audio, samplerate=model.config.sampling_rate)
sd.wait()