import sounddevice as sd

# 这里填你想要测试的设备名称
# 注意：为了查出【物理硬件】的真实通道数，请测试直接访问硬件的名字，而不是 default。
# 比如通过 python -m sounddevice 查到的带 hw 字样的原名。
device_name = "ReSpeaker 4 Mic Array"  

print(f"===== 正在探测设备: '{device_name}' =====")

# 1. 直接查询设备声明的最大输入通道数
try:
    dev_info = sd.query_devices(device_name, kind='input')
    max_ch = dev_info['max_input_channels']
    print(f"📢 硬件信息宣称的最大录音通道数: {max_ch}\n")
except Exception as e:
    print(f"查询设备信息失败: {e}\n")

# 2. 用 check_input_settings 挨个试探 (测试 1 到 10 通道)
print("开始暴力试探具体支持的通道数（采样率假设为 16000）...")
supported_channels =[]

for ch in range(1, 11):
    try:
        # 如果不报错，说明支持
        sd.check_output_settings(
            device=device_name, 
            channels=ch, 
            samplerate=16000  # 可以根据你的设备尝试 16000 或 48000
        )
        supported_channels.append(ch)
        print(f"✅ 成功：完全支持 {ch} 通道录音")
    except Exception as e:
        # 如果报错，说明不支持，我们可以把报错信息稍微精简一下打印出来
        err_msg = str(e).split('\n')[0] 
        print(f"❌ 失败：不支持 {ch} 通道  (原因: {err_msg})")

print(f"\n🎯 最终探测结果: 该设备支持的输入通道数有: {supported_channels}")
