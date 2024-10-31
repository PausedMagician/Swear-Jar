import asyncio
import os
from threading import Thread
import time
from datetime import datetime
import sounddevice as sd
import scipy.io.wavfile
import whisper
import json
import regex

# import torch
# torch.zeros(1).cuda()


stopped = False
fs = 44100  # Sample rate
seconds = 5  # Duration of recording


swearRegexTemplate = "(?: |^)({baseword}({possible_endings}){0,1})(?: |$|\.|\?|!)"

# File exists checks
if not os.path.exists("temps"):
    os.makedirs("temps")
if not os.path.exists("data/words.json"):
    print("Error: words.json not found!")
    exit(1)
if not os.path.exists("data/log.txt"):
    with open("data/log.txt", "w") as f:
        f.write("Log file created\n")
if not os.path.exists("data/spoken.json"):
    with open("data/spoken.json", "w") as f:
        f.write("{}")



async def write(name, fs, myrecording):
    scipy.io.wavfile.write(name, fs, myrecording)  # Save as WAV file 
    whisperer.addToQueue(name)

def record():
    global stopped
    while(not stopped):
        # print("Recording...")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        # print("Recording done")
        name = "temps/output" + str(int(time.time())) + ".wav"
        asyncio.run(write(name, fs, myrecording))  # Save as WAV file

recordingThread = Thread(target=record)



class Whisperer:
    # static variables
    instance = None
    # static variables
    queue = []

    def __init__(self):
        global instance, queue
        # base.en, small.en, medium.en, turbo
        self.model = whisper.load_model('turbo', device='cuda')
        self.sweardata = []
        if not os.path.exists("data/words.json"):
            print("Error: words.json not found!")
            exit(1)
        with open("data/words.json") as f:
            wordData = json.load(f)
            for word in wordData:
                if not "possible_endings" in word:
                    self.sweardata.append(swearRegexTemplate.replace("{baseword}", word["baseword"]).replace("{possible_endings}", ""))
                else:
                    self.sweardata.append(swearRegexTemplate.replace("{baseword}", word["baseword"]).replace("{possible_endings}", "|".join(word["possible_endings"])))
        print("Words loaded:\n\t" + "\n\t".join(self.sweardata))
        self.allSpoken = open("data/log.txt", "a")
        instance = self
        queue = []
        
    @staticmethod
    def getInstance():
        return instance
    
    @staticmethod
    def getQueue():
        return queue
    @staticmethod
    def addToQueue(item):
        queue.append(item)
    
    def transcribe(self, audioPath):
        result = self.model.transcribe(audioPath)
        print("Heard:\n" + result["text"])
        self.allSpoken.write(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\t{result["text"]}\n")
        matches = []
        for swearRegex in self.sweardata:
            reg = regex.Regex(swearRegex)
            matches += reg.findall(result["text"])
        return matches
    
    def step(self):
        if len(queue) > 0:
            item = queue.pop(0)
            res = self.transcribe(item)
            # delete the file
            os.remove(item)
            return res
        return None


spokenWords = []
with open("data/spoken.json") as f:
    spokenWords = json.load(f)

whisperer = Whisperer()

print("Listening...")
recordingThread.daemon = True
recordingThread.start()

try:
    while(not stopped):
        while (res := whisperer.step()) is not None:
            if res != []:
                print("Swear detected!")
                for match in res:
                    word = match[0]
                    print(word)
                    if word in spokenWords:
                        print("Spoken word detected!")
                        spokenWords[word] += 1
                    else:
                        print("New word detected!")
                        spokenWords[word] = 1
        time.sleep(1)
except KeyboardInterrupt:
    whisperer.allSpoken.close()
    with open("data/spoken.json", "w") as f:
        json.dump(spokenWords, f)
    stopped = True
    print("Goodbye!")
    exit(0)