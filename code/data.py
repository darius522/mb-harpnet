from sys import path_importer_cache
import torch
import torch.utils.data
import random
import torch
import tqdm
import soundfile as sf
from pathlib import Path
import os
import random

from scipy.signal.windows import *
import torchaudio.transforms as T


from utils import soundfile_info, band_limited_noise

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_in,
        root_out1,
        root_out2,
        split='tr',
        chunk_size=None,
        random_chunks=False,
        sample_rate=32000,
        num_chan=1,
        scaler=1.0,
        training=False
    ):

        self.root_in = root_in
        self.root_out1 = root_out1
        self.root_out2 = root_out2
        self.split = split
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_size)
        self.random_chunks = random_chunks
        self.num_chan = num_chan
        self.scaler = scaler
        self.training = training
        # set the input and output files (accept glob)
        self.paths = list(self._get_paths())
        if not self.paths:
            raise RuntimeError("Dataset is empty, please check parameters")
        
        self.cos = 1.0 - cosine(self.chunk_size//4)
        
        self.downsampling = T.Resample(sample_rate, sample_rate//2, dtype=torch.float32)
        self.upsampling = T.Resample(sample_rate//2, sample_rate, dtype=torch.float32)

    def __getitem__(self, index):
        path_in, path_out1, path_out2 = self.paths[index]
        info_in = soundfile_info(path_in)
        info_out1 = soundfile_info(path_out1)
        info_out2 = soundfile_info(path_out2)
        if self.random_chunks:
            duration = min(info_in.samples, info_out1.samples, info_out2.samples)
            start = int(random.uniform(0, duration - self.chunk_size))
            stop = start+self.chunk_size
        else:
            start = 0
            stop = None

        # load actual audio
        audio_in, sr1 = sf.read(
            path_in,
            always_2d=True,
            start=start,
            stop=stop,
        )
        audio_out1, sr2 = sf.read(
            path_out1,
            always_2d=True,
            start=start,
            stop=stop,
        )
        
        audio_out2, sr2 = sf.read(
            path_out2,
            always_2d=True,
            start=start,
            stop=stop,
        )
        
        assert (self.sample_rate == sr1 and self.sample_rate == sr2), 'Incompatible SR found!'
        
        audio_in_t = torch.tensor(audio_in, dtype=torch.float32)
        audio_out1_t = torch.tensor(audio_out1, dtype=torch.float32)
        audio_out2_t = torch.tensor(audio_out2, dtype=torch.float32)
        if self.num_chan == 1:
            audio_in_t = audio_in_t.mean(1, keepdim=True)
            audio_out1_t = audio_out1_t.mean(1, keepdim=True)
            audio_out2_t = audio_out2_t.mean(1, keepdim=True)
        
        #audio_in_t   = audio_in_t.T
        audio_out1_t = self.downsampling(audio_out1_t.T)
        audio_out2_t = audio_out2_t.T
        
        tmp = self.upsampling(audio_out1_t) 
        mn = min(tmp.shape[-1], audio_out2_t.shape[-1])
        audio_in_t = tmp[...,:mn] + audio_out2_t[...,:mn]

        return audio_in_t, audio_out1_t[...,:mn], audio_out2_t[...,:mn]

    def __len__(self):
        return len(self.paths)

    def _get_paths(self):
        """Loads track"""

        p1 = Path(self.root_in, self.split)
        for i, track_path_in in tqdm.tqdm(enumerate(p1.iterdir())):
            track_path_out1 = os.path.join(self.root_out1, self.split, os.path.basename(track_path_in))
            track_path_out2 = os.path.join(self.root_out2, self.split, os.path.basename(track_path_in))
            if track_path_in and track_path_out1 and track_path_out2 and self.chunk_size is not None:
                info_in = soundfile_info(track_path_in)
                info_out1 = soundfile_info(track_path_out1)
                info_out2 = soundfile_info(track_path_out2)
                # get metadata
                if (not info_in.samplerate == self.sample_rate and\
                    not info_out1.samplerate == self.sample_rate and\
                    not info_out2.samplerate == self.sample_rate):
                    print("Exclude track due to different sample rate ", track_path_in.stem)
                    continue
                if info_in.samples > self.chunk_size and info_out1.samples > self.chunk_size and info_out2.samples > self.chunk_size:
                    yield track_path_in, track_path_out1, track_path_out2
            else:
                yield track_path_in, track_path_out1, track_path_out2