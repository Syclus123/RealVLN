import sounddevice as sd

device_name="Jabra Link 370: USB Audio (hw:0,0)"
device_name="ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)"

def get_device_by_exact_name(exact_name, kind='any'):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if kind == 'input' and dev['max_input_channels'] == 0:
            continue
        if kind == 'output' and dev['max_output_channels'] == 0:
            continue
        print(dev["name"])
        if dev['name'] == exact_name:
            return i
    raise ValueError(f"未找到名称完全匹配 '{exact_name}' 的设备")

device_id = get_device_by_exact_name(device_name)  # 替换成你的设备ID
#device_id = 0
samplerates_to_test = [8000, 16000, 22050, 44100, 48000, 96000, 192000]

print(f"测试设备 {device_id}: {sd.query_devices(device_id)['name']}")
for sr in samplerates_to_test:
    try:
        sd.check_input_settings(device=device_id, samplerate=sr)
        print(f"✅ 支持采样率: {sr} Hz")
    except Exception as e:
        print(f"❌ 不支持 {sr} Hz: {e}")
