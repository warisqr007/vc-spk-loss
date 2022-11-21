import resampy
import torch
#from .audio_utils import MAX_WAV_VALUE, mel_spectrogram, normalize
from scipy.io.wavfile import read
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import os, sys
from speaker_encoder.voice_encoder import SpeakerEncoder
from speaker_encoder.audio import preprocess_wav
from pathlib import Path
import numpy as np
from os.path import join, basename, split
from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor 
from functools import partial
import glob
import glob2
import argparse

MAX_WAV_VALUE = 32768.0

mel_basis = {}
hann_window = {}

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram(
    y, 
    n_fft=1024, 
    num_mels=80, 
    sampling_rate=24000, 
    hop_size=240, 
    win_size=1024, 
    fmin=0, 
    fmax=8000, 
    center=False,
    output_energy=False,
):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    mel_spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    mel_spec = spectral_normalize_torch(mel_spec)
    if output_energy:
        energy = torch.norm(spec, dim=1)
        return mel_spec, energy
    else:
        return mel_spec

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def compute_mel(wav_path):
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
        return melspec.squeeze(0).T.unsqueeze(0)


def build_from_path(in_dir, out_dir, weights_fpath, num_workers=1):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wavfile_paths = glob2.glob(f"{in_dir}/**/*mic2.wav") #glob.glob(os.path.join(in_dir, '/**/*.wav'))
    print(f"Globbed {len(wavfile_paths)} wave files.")
    wavfile_paths= sorted(wavfile_paths)
    for wav_path in wavfile_paths:
        futures.append(executor.submit(
            partial(_compute_spkEmbed, out_dir, wav_path, weights_fpath)))
    return [future.result() for future in tqdm(futures)]

def _compute_spkEmbed(out_dir, wav_path, weights_fpath):
    utt_id = os.path.basename(wav_path).rstrip(".wav")
    fpath = Path(wav_path)

    mel = compute_mel(fpath)
    encoder = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder', verbose=False)
    embed = encoder(mel)
    embed = embed.squeeze(0).detach().numpy().shape
    #wav = preprocess_wav(fpath)

    #encoder = SpeakerEncoder(weights_fpath)
    #embed = encoder.embed_utterance(wav)
    foldr = wav_path.split('/')[-3]
    out_dir = f'{out_dir}/{foldr}'
    os.makedirs(out_dir, exist_ok=True)
    fname_save = os.path.join(out_dir, f"{utt_id}.npy")
    np.save(fname_save, embed, allow_pickle=False)
    return os.path.basename(fname_save)

def preprocess(in_dir, out_dir, weights_fpath, num_workers):
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir, weights_fpath, num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--in_dir', type=str, 
    #     default='/mnt/data2/bhanu/datasets/all_data_for_ac_vc')
    parser.add_argument('--in_dir', type=str, 
        default='/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/')
    parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--out_dir_root', type=str, 
    #     default='/mnt/data2/bhanu/datasets/dvec')
    parser.add_argument('--out_dir_root', type=str, 
        default='/mnt/data1/waris/model_preprocessing/transformer-vc-vctk/dvec-SSE/')
    parser.add_argument('--spk_encoder_ckpt', type=str, \
        default='speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    args = parser.parse_args()
    
    split_list = ['train-clean-100', 'train-clean-360']

    # sub_folder_list = os.listdir(args.in_dir)
    # sub_folder_list.sort()
    
    args.num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print("Number of workers: ", args.num_workers)
    ckpt_step = os.path.basename(args.spk_encoder_ckpt).split('.')[0].split('_')[-1]
    spk_embed_out_dir = args.out_dir_root #os.path.join(args.out_dir_root, f"GE2E_spkEmbed_step_{ckpt_step}")
    print("[INFO] spk_embed_out_dir: ", spk_embed_out_dir)
    os.makedirs(spk_embed_out_dir, exist_ok=True)

    # for data_split in split_list:
        # sub_folder_list = os.listdir(args.in_dir, data_split) 
        # for spk in sub_folder_list:
            # print("Preprocessing {} ...".format(spk))
            # in_dir = os.path.join(args.in_dir, dataset, spk)
            # if not os.path.isdir(in_dir): 
                # continue
            # # out_dir = os.path.join(args.out_dir, spk)
            # preprocess(in_dir, spk_embed_out_dir, args.spk_encoder_ckpt, args.num_workers)
    # for data_split in split_list:
    #     in_dir = os.path.join(args.in_dir, data_split)
    #     preprocess(in_dir, spk_embed_out_dir, args.spk_encoder_ckpt, args.num_workers)
    preprocess(args.in_dir, spk_embed_out_dir, args.spk_encoder_ckpt, args.num_workers)

    print("DONE!")
    sys.exit(0)



