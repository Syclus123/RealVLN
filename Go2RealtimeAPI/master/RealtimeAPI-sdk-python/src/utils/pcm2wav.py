import wave
import os
import sys 

raw_path, output_path = sys.argv[1:]

with open(raw_path, 'rb') as raw:
    pcm_data = raw.read()

with wave.open(output_path, "wb") as wf: 
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(pcm_data)