import librosa
from pydub import AudioSegment
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import json

def load_audio_file(file_path):
    audio = AudioSegment.from_file(file_path)
    audio_samples = audio.get_array_of_samples()
    audio_samples = np.array(audio_samples, dtype=np.float32)
    return audio_samples, audio.frame_rate

input_dir = "/path/to/Dataset/wav"
output_dir = "/path/to/Dataset/music_features"
entries = sorted(os.listdir(input_dir))
pitch_scaler = StandardScaler()
loud_scaler = StandardScaler()
chroma_scaler = StandardScaler()
spectral_scaler = StandardScaler()

def normalize(in_dir, means, stds):
    names = ['pitch.npy', 'loudness.npy', 'spectral_centroid.npy', 'chroma.npy']
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    mins = [min_value, min_value, min_value, min_value]
    maxs = [max_value, max_value, max_value, max_value]
    for filename in os.listdir(in_dir):
        filename = os.path.join(in_dir, filename)
        data = np.load(filename)
        values = [None] * 4
        for i in range(4):
            values[i] = (data[names[i]] - means[i]) / stds[i]
            maxs[i]= max(maxs[i], np.max(values[i]))
            mins[i] = min(mins[i], np.min(values[i]))
        np.savez(
            filename,
            **{"loudness": values[1], "pitch": values[0], "chroma": values[3], "spectral_centroid": values[2]},
        )

    return mins, maxs

for path in tqdm(entries):
    if path.endswith(".wav") and os.path.isfile(os.path.join(input_dir, path)):
        audio_samples, frame_rate = load_audio_file(os.path.join(input_dir, path))
        FRAMES = 1 
        hop_length = int(round(frame_rate / FRAMES))  # 计算每秒30帧所需的hop_length  
        n_fft = 2048 
        # 计算响度
        loudness = librosa.feature.rms(y=audio_samples, frame_length=n_fft, hop_length=hop_length)
        # 计算基频  
        pitches, _ = librosa.piptrack(y=audio_samples, sr=frame_rate, n_fft=n_fft,  
                                    hop_length=hop_length)  
        # 计算音调  
        chroma = librosa.feature.chroma_stft(y=audio_samples, sr=frame_rate, n_fft=n_fft,  hop_length=hop_length)
        # 计算谱质心  
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_samples, sr=frame_rate, n_fft=n_fft, hop_length=hop_length)  
        data = {  
            'loudness': loudness[0],  
            'pitch': pitches,  
            'chroma': chroma,  
            # 'mfcc': mfcc,  
            'spectral_centroid': spectral_centroid[0],  
            # 'spectral_rolloff': spectral_rolloff[0],  
            # 'zero_crossing_rate': zero_crossing_rate[0]  
        }
        base = os.path.splitext(path)[0]
        np_path = os.path.join(output_dir, base + ".npz")
        
        loud = loudness[0]
        spc = spectral_centroid[0]
        
        pitch_scaler.partial_fit(pitches.reshape((-1, 1)))
        loud_scaler.partial_fit(loud.reshape((-1, 1)))
        spectral_scaler.partial_fit(spc.reshape((-1, 1)))
        chroma_scaler.partial_fit(chroma.reshape((-1, 1)))
        # loud = (loud - loud.mean())  / loud.std()
        # spc = (spc - spc.mean())  / spc.std()
        # pitches = (pitches - pitches.mean())  / pitches.std()
        # chroma = (chroma - chroma.mean())  / chroma.std()
        
        np.savez(
            np_path,
            **{"loudness": loud, "pitch": pitches, "chroma": chroma, "spectral_centroid": spc},
        )
print("Computing statistic quantities ...")
# Perform normalization if necessary
pitch_mean = pitch_scaler.mean_[0]
pitch_std = pitch_scaler.scale_[0]
loudness_mean = loud_scaler.mean_[0]
loudness_std = loud_scaler.scale_[0]
spectral_mean = spectral_scaler.mean_[0]
spectral_std = spectral_scaler.scale_[0]
chroma_mean = chroma_scaler.mean_[0]
chroma_std = chroma_scaler.scale_[0]

means = [pitch_mean, loudness_mean, spectral_mean, chroma_mean]
stds = [pitch_std, loudness_std, spectral_std, chroma_std]

mins, maxs = normalize(output_dir, means, stds)

pitch_min, loudness_min, spectral_min, chroma_min = mins
pitch_max, loudness_max, spectral_max, chroma_max = maxs

with open(os.path.join(output_dir, "stats.json"), "w") as f:
    stats = {
        "pitch": [
            float(pitch_min),
            float(pitch_max),
            float(pitch_mean),
            float(pitch_std),
        ],
        "loudness": [
            float(loudness_min),
            float(loudness_max),
            float(loudness_mean),
            float(loudness_std),
        ],
        "spectral_centroid": [
            float(spectral_min),
            float(spectral_max),
            float(spectral_mean),
            float(spectral_std),
        ],
        "chroma": [
            float(chroma_min),
            float(chroma_max),
            float(chroma_mean),
            float(chroma_std),
        ]
    }
    f.write(json.dumps(stats))
