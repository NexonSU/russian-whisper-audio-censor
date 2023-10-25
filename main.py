import whisper
import io
import re
from pydub import AudioSegment
from pydub.generators import Sine

word_filter = io.open("word_filter.txt", mode="r", encoding="utf-8").read().splitlines()

model = whisper.load_model("large")
result = model.transcribe(audio="C:/Users/NexonSU/Downloads/zombak_001_1/zombak_001_1.wav", word_timestamps=True)

a = AudioSegment.from_wav("C:/Users/NexonSU/Downloads/zombak_001_1/zombak_001_1.wav")
parts = []
begin = 0

regex = re.compile('[\W+]')

for segment in result["segments"]:
    for word in segment["words"]:
        word["word"] = regex.sub('', word["word"])
        if word["word"].removeprefix(" ") in word_filter:
            print("!!!", word["word"], word["start"])

            # keep sound before silence
            s = a[begin*1000:word["start"]*1000]
            parts.append(s)  
            
            # create silence
            duration = (word["end"] - word["start"]) * 1000
            #s = AudioSegment.silent(duration)
            s = Sine(1000).to_audio_segment(duration).apply_gain(-20)
            parts.append(s)

            # value for next loop
            begin = word["end"]

parts.append(a[begin*1000:])

# join all parts using standard `sum()` but it need `parts[0]` as start value   
b = sum(parts[1:], parts[0])

# save it
b.export('C:/Users/NexonSU/Downloads/zombak_001_1/new_audio.wav', format='wav')