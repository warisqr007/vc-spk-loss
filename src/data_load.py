import random
import numpy as np
import torch
import os
from collections import OrderedDict
from utils.f0_utils import get_cont_lf0, convert_continuous_f0
import resampy
from .audio_utils import MAX_WAV_VALUE, load_wav, mel_spectrogram, normalize


def read_fids(fid_list_f):
    with open(fid_list_f, 'r') as f:
        fids = [l.strip().split()[0] for l in f if l.strip()]
    return fids


class MultiSpkVcCollate():
    """Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step=1, give_uttids=False,
                use_spk_dvec=False):
        self.n_frames_per_step = n_frames_per_step
        self.give_uttids = give_uttids
        self.use_spk_dvec = use_spk_dvec

    def __call__(self, batch):
        batch_size = len(batch)              
        # Prepare different features 
        # (ppg, mel, spk_dvec, fid, lwav)
        ppgs = [x[0] for x in batch]
        mels = [x[1] for x in batch]
        fids = [x[-2] for x in batch]
        lwav = [x[-1] for x in batch]
        maxlwav = max(lwav)
        # print(f'torch batch length : {batch[0]}')
        spk_ids = [x[2] for x in batch]
        spk_ids = torch.stack(spk_ids).float()
        # if len(batch[0]) == 5:
        #     spk_ids = [x[2] for x in batch]
        #     if self.use_spk_dvec:
        #         # use d-vector
        #         spk_ids = torch.stack(spk_ids).float()
        #     else:
        #         # use one-hot ids
        #         spk_ids = torch.LongTensor(spk_ids)
        # Pad features into chunk
        ppg_lengths = [x.shape[0] for x in ppgs]
        mel_lengths = [x.shape[0] for x in mels]
        max_ppg_len = max(ppg_lengths)
        max_mel_len = max(mel_lengths)
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += (self.n_frames_per_step - max_mel_len % self.n_frames_per_step)
        ppg_dim = ppgs[0].shape[1]
        mel_dim = mels[0].shape[1]
        ppgs_padded = torch.FloatTensor(batch_size, max_ppg_len, ppg_dim).zero_()
        mels_padded = torch.FloatTensor(batch_size, max_mel_len, mel_dim).zero_()
        stop_tokens = torch.FloatTensor(batch_size, max_mel_len).zero_()
        for i in range(batch_size):
            cur_ppg_len = ppgs[i].shape[0]
            cur_mel_len = mels[i].shape[0]
            ppgs_padded[i, :cur_ppg_len, :] = ppgs[i]
            mels_padded[i, :cur_mel_len, :] = mels[i]
            stop_tokens[i, cur_ppg_len-self.n_frames_per_step:] = 1
        if False:
            print('hererererere')
            ret_tup = (ppgs_padded, mels_padded, torch.LongTensor(ppg_lengths), \
                torch.LongTensor(mel_lengths), spk_ids, stop_tokens,torch.IntTensor(maxlwav))
            if self.give_uttids:
                return ret_tup + (fids, )
            else:
                return ret_tup
        else:
            ret_tup = (ppgs_padded, mels_padded, torch.LongTensor(ppg_lengths), \
                torch.LongTensor(mel_lengths), spk_ids, stop_tokens, torch.LongTensor([maxlwav]))
            if self.give_uttids:
                return ret_tup + (fids, )
            else:
                return ret_tup


class VCTKDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_file: str,
        ppg_dir: str,
        #f0_dir: str,
        wav_dir: str,
        spk_dvec_dir: str,
        min_max_norm_mel: bool = False,
        mel_min: float = None,
        mel_max: float = None,
        ppg_file_ext: str = "ling_feat.npy",
        #f0_file_ext: str = "f0.npy",
        wav_file_ext: str = "wav",
    ):
        self.fid_list = read_fids(meta_file)
        self.ppg_dir = ppg_dir
        #self.f0_dir = f0_dir
        self.wav_dir = wav_dir
        self.spk_dvec_dir = spk_dvec_dir

        self.ppg_file_ext = ppg_file_ext
        #self.f0_file_ext = f0_file_ext
        self.wav_file_ext = wav_file_ext

        self.min_max_norm_mel = min_max_norm_mel
        if min_max_norm_mel:
            print("[INFO] Min-Max normalize Melspec.")
            assert mel_min is not None
            assert mel_max is not None
            self.mel_max = mel_max
            self.mel_min = mel_min
        
        random.seed(1234)
        random.shuffle(self.fid_list)
        print(f'[INFO] Got {len(self.fid_list)} samples.')
        
    def __len__(self):
        return len(self.fid_list)
    
    def get_spk_dvec(self, fid):
        spk_dvec_path = f"{self.spk_dvec_dir}/{fid}.npy"
        return torch.from_numpy(np.load(spk_dvec_path))
    
    # def get_ppg_input(self, fid): #ppg-ERMS-arctic_a0343.npy
    #     sprf , wfle = fid.split('/')
    #     ppg = np.load(f"{self.arctic_ppg_dir}/ppg-{sprf}-{wfle}.{self.ppg_file_ext}")
    #     return ppg
    
    def compute_mel(self, wav_path):
        audio, sr = load_wav(wav_path)
        lwav = len(audio)
        if sr != 24000:
            audio = resampy.resample(audio, sr, 24000)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        audio = torch.FloatTensor(audio).unsqueeze(0)
        melspec = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=80,
            sampling_rate=24000,
            hop_size=240,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )
        return melspec.squeeze(0).numpy().T, lwav

    def bin_level_min_max_norm(self, melspec):
        # frequency bin level min-max normalization to [-4, 4]
        mel = (melspec - self.mel_min) / (self.mel_max - self.mel_min) * 8.0 - 4.0
        return np.clip(mel, -4., 4.)   

    def __getitem__(self, index):
        fid = self.fid_list[index]
        
        # 1. Load features
        sprf , wfle = fid.split('/')
        ppg = np.load(f"{self.ppg_dir}/{fid}.{self.ppg_file_ext}")
        #ppg = self.get_ppg_input(fid)
        #f0 = np.load(f"{self.f0_dir}/{fid}.{self.f0_file_ext}")
        mel, lwav = self.compute_mel(f"{self.wav_dir}/{sprf}/wav/{wfle}.{self.wav_file_ext}")
        if self.min_max_norm_mel:
            mel = self.bin_level_min_max_norm(mel)
        
        ppg, mel = self._adjust_lengths(ppg, mel)
        spk_dvec = self.get_spk_dvec(fid)

        # 2. Convert f0 to continuous log-f0 and u/v flags
        #uv, cont_lf0 = get_cont_lf0(f0, 10.0, False)
        # cont_lf0 = (cont_lf0 - np.amin(cont_lf0)) / (np.amax(cont_lf0) - np.amin(cont_lf0))
        # cont_lf0 = self.utt_mvn(cont_lf0)
        #lf0_uv = np.concatenate([cont_lf0[:, np.newaxis], uv[:, np.newaxis]], axis=1)

        # uv, cont_f0 = convert_continuous_f0(f0)
        # cont_f0 = (cont_f0 - np.amin(cont_f0)) / (np.amax(cont_f0) - np.amin(cont_f0))
        # lf0_uv = np.concatenate([cont_f0[:, np.newaxis], uv[:, np.newaxis]], axis=1)
        
        # 3. Convert numpy array to torch.tensor
        ppg = torch.from_numpy(ppg)
        #lf0_uv = torch.from_numpy(lf0_uv)
        mel = torch.from_numpy(mel)
        return (ppg, mel, spk_dvec, fid, lwav)

    # def check_lengths(self, ppg, mel):
    #     LEN_THRESH = 10
    #     assert abs(len(ppg) - len(f0)) <= LEN_THRESH, \
    #         f"{abs(len(ppg) - len(f0))}"
    #     assert abs(len(mel) - len(f0)) <= LEN_THRESH, \
    #         f"{abs(len(mel) - len(f0))}"
    
    def _adjust_lengths(self, ppg, mel):
        #self.check_lengths(f0, ppg, mel)
        min_len = min(
            len(ppg),
            len(mel),
        )
        ppg = ppg[:min_len]
        mel = mel[:min_len]
        return ppg, mel