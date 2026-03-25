import pyaudio
import wave

print("🎤 准备录音 5 秒钟...")
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1600)

print("🔴 开始！请大声喊几句 '狗子跟上'...")
frames = [stream.read(1600, exception_on_overflow=False) for _ in range(0, int(16000 / 1600 * 5))]

stream.stop_stream()
stream.close()
audio.terminate()

with wave.open("test.wav", 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
print("✅ 已保存为 test.wav")
