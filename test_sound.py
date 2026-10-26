import math
import os
import struct
import tempfile
import time
import wave

from usb_sound_controller import USB_SoundController


def make_tone(path, freq=440, seconds=1.0, rate=44100):
    # Writes a sine-wave WAV, so the test needs no sound files.
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(int(seconds * rate)))
        w.writeframes(frames)


tmp = tempfile.mkdtemp()
low, high = os.path.join(tmp, "low.wav"), os.path.join(tmp, "high.wav")
make_tone(low, 440)
make_tone(high, 880)

sound = USB_SoundController(volume=0.4)
try:
    print("1. Tone: you should hear a low beep then a high beep")
    sound.play_audio(low)
    time.sleep(1.5)
    sound.play_audio(high)
    time.sleep(1.5)

    print("2. Volume: three beeps, quiet -> loud")
    for vol in (0.2, 0.5, 1.0):
        sound.set_volume(vol)
        sound.play_audio(low)
        time.sleep(1.5)

    print("3. Text to speech: you should hear a voice")
    sound.set_volume(0.7)
    sound.play_text_to_speech("Sound Test. Sound Test. Sound Test.")
    time.sleep(6)

    print("4. Stop: a long beep should cut off after about half a second")
    make_tone(low, 440, seconds=4.0)
    sound.play_audio(low)
    time.sleep(0.5)
    sound.stop_sound()
    time.sleep(4)  # stay quiet for the rest of the tone, so you can tell it was cut off
finally:
    sound.close()
    if os.path.exists("temp_tts.wav"):
        os.remove("temp_tts.wav")
