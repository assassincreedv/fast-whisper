from faster_whisper import WhisperModel,BatchedInferencePipeline
import sys
import os
import time

def transcribe(audio_path):
    if not os.path.isfile(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return

    total_start = time.time()

    # Step 1: 加载模型
    model_start = time.time()
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")

    batched_model = BatchedInferencePipeline(model=model)
    model_end = time.time()
    print(f"✅ 模型加载耗时: {model_end - model_start:.2f} 秒")

    # Step 2: 开始转录
    transcribe_start = time.time()
    segments, info = batched_model.transcribe(audio=audio_path, beam_size=5, batch_size=16, word_timestamps=True)
    transcribe_end = time.time()
    print(f"🕒 音频转录耗时: {transcribe_end - transcribe_start:.2f} 秒")

    # Step 3: 输出结果
    print(f"\n🔍 检测语言: {info.language}")
    print("\n📋 转录内容：\n")
    for segment in segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

    total_end = time.time()
    print(f"\n🚀 总耗时: {total_end - total_start:.2f} 秒")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 transcribe_audio.py <音频路径>")
    else:
        transcribe(sys.argv[1])
