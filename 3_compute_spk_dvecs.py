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
    wav = preprocess_wav(fpath)

    encoder = SpeakerEncoder(weights_fpath)
    embed = encoder.embed_utterance(wav)
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
    parser.add_argument('--num_workers', type=int, default=4)
    # parser.add_argument('--out_dir_root', type=str, 
    #     default='/mnt/data2/bhanu/datasets/dvec')
    parser.add_argument('--out_dir_root', type=str, 
        default='/mnt/data1/waris/model_preprocessing/transformer-vc-vctk/dvec')
    parser.add_argument('--spk_encoder_ckpt', type=str, \
        default='speaker_encoder/ckpt/pretrained_bak_5805000.pt')

    args = parser.parse_args()
    
    split_list = ['train-clean-100', 'train-clean-360']

    # sub_folder_list = os.listdir(args.in_dir)
    # sub_folder_list.sort()
    
    args.num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print("Number of workers: ", args.num_workers)
    ckpt_step = os.path.basename(args.spk_encoder_ckpt).split('.')[0].split('_')[-1]
    spk_embed_out_dir = os.path.join(args.out_dir_root, f"GE2E_spkEmbed_step_{ckpt_step}")
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



