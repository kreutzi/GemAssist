import google.generativeai as genai
from search import search_youtube
import os
import numpy as np
from openwakeword.model import Model
from sys import byteorder
from array import array
from struct import pack
import re
import pyaudio
import wave
from faster_whisper import WhisperModel
from openwakeword.model import Model
import whisper
import shlex  # Import shlex module for proper shell quoting
import wave
from pydub import AudioSegment
from pydub.playback import play
from piper.voice import PiperVoice

model_size = "medium.en"
whispmodel = WhisperModel(model_size, device="cuda", compute_type="float16")
owwModel = Model(enable_speex_noise_suppression=True)

piperemodel = 'n_GB-alba-medium.onnx'

def tts_command(text):
  voice = PiperVoice.load(piperemodel)
  wav_file = wave.open('output.wav', 'w')
  text=re.sub(r'\*', '', text)
  text='"'+shlex.quote(text)+'"'
  text=text.replace("\n","")
  voice.synthesize(text,wav_file)
  wav_file.close()
  play(AudioSegment.from_wav("output.wav"))

THRESHOLD = 500
# CHUNK_SIZE = 1024
CHUNK_SIZE = 1280
FORMAT = pyaudio.paInt16
# RATE = 44100
RATE = 16000

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)
    while 1:
        aud = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)

        # Feed to openWakeWord model
        
        prediction = owwModel.predict(aud,threshold={'hey_jarvis': 0.9},debounce_time=2)
        if prediction['hey_jarvis'] >= 0.9:
            print("Wakeword detected!")
            print("please speak a word into the microphone")
            break
    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)
        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



API_KEY = "XXXX-"
genai.configure(api_key=API_KEY)

def program_exit():
    """if the user says 'exit', the program will exit."""
    exit()

def youtube_search(query:str):
    """searches for a query on YouTube and opens the most relevent result in the browser."""
    global response
    urls = search_youtube(query)
    response=chat.send_message(f"results are {urls} . open the most relevant url using browser. your response should only be 'Opened (title of opened url from results list) in youtube'")
    return(response.text)
    
def browser_open(url:str):
    """Opens a url in the browser. and returns 'opened'."""
    os.system(f"google-chrome-stable --new-window {url}")
    return f"opened"
              
model = genai.GenerativeModel(model_name='gemini-1.5-pro',tools=[program_exit, browser_open, youtube_search])
chat = model.start_chat(enable_automatic_function_calling=True,history=[])
response=chat.send_message("You are Jarvis, a capable and helpful assistant. Start the conversation with a friendly but professional greeting. Keep your responses concise and helpful. You are designed to assist with tasks and provide information – focus on those abilities. Let's get started!")

while 1:
    record_to_file('demo.wav')
    segments, info = whispmodel.transcribe("demo.wav",language="en", beam_size=5, word_timestamps=True, vad_filter=True)
    tx=''
    for segment in segments:
        tx+=segment.text
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    user_input=tx
    tx+=""
    response = chat.send_message(user_input)
    print(response.text)
    tts_command(response.text)
