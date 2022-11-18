import glob2
import random
import numpy as np
from sklearn.model_selection import train_test_split

# train_list = "/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/SV2TTS/synthesizer/train_split.txt"
# dev_list = "/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/SV2TTS/synthesizer/dev_split.txt"

# with open(train_list, encoding="utf-8") as f:
#     train_metadata = [line.strip().split("|") for line in f]

# with open('train.txt', mode='wt', encoding='utf-8') as myfile:
#     for idx in range(len(train_metadata)):
#         _, spkr, fid = train_metadata[idx][0].split("-")
#         fid = fid.split(".")[0]
#         myfile.write(f'{spkr}/{fid}')
#         myfile.write('\n')

# with open(dev_list, encoding="utf-8") as f:
#     dev_metadata = [line.strip().split("|") for line in f]
# with open('dev.txt', mode='wt', encoding='utf-8') as myfile:
#     for idx in range(len(dev_metadata)):
#         _, spkr, fid = dev_metadata[idx][0].split("-")
#         fid = fid.split(".")[0]
#         myfile.write(f'{spkr}/{fid}')
#         myfile.write('\n')

# Test Subjects
# p225  23  F    English    Southern  England
# p226  22  M    English    Surrey
# p227  38  M    English    Cumbria
# p228  22  F    English    Southern  England
# p229  23  F    English    Southern  England
# p232  23  M    English    Southern  England

wav_file_list = glob2.glob(f"/mnt/data1/waris/datasets/vctk/wav48_silence_trimmed/**/*mic2.wav")
test_speaker_list = ['p225', 'p226', 'p227', 'p228', 'p229', 'p232']

ids = []
for t in wav_file_list:
    spkr = t.split('.')[0].split('/')[-3]
    fid = t.split('.')[0].split('/')[-1]
    uid = int(fid.split('_')[1])
    # with open('/path/to/filename.txt', mode='wt', encoding='utf-8') as myfile:
    if spkr not in test_speaker_list and uid <= 330:
        ids.append(f'{spkr}/{fid}')

ids = np.array(ids)
np.random.shuffle(ids)

data_train, data_test, labels_train, labels_test = train_test_split(ids, ids, test_size=0.05, random_state=42)

with open('train.txt', mode='wt', encoding='utf-8') as myfile:
    for s in data_train:
        myfile.write(s)
        myfile.write('\n')
with open('dev.txt', mode='wt', encoding='utf-8') as myfile:
    for s in data_test:
        myfile.write(s)
        myfile.write('\n')

# import glob2
# import random
# import numpy as np
# from sklearn.model_selection import train_test_split

# wav_file_list = glob2.glob(f"/mnt/data1/waris/datasets/data/arctic_dataset/all_data_for_ac_vc_train/**/*.wav")

# ids = []
# for t in wav_file_list:
#     spkr = t.split('.')[0].split('/')[-3]
#     fid = t.split('.')[0].split('/')[-1]
#     wav  = t.split('.')[0].split('/')[-2]
#     # with open('/path/to/filename.txt', mode='wt', encoding='utf-8') as myfile:

#     ids.append(f'{spkr}/{fid}')

# ids = np.array(ids)
# np.random.shuffle(ids)

# data_train, data_test, labels_train, labels_test = train_test_split(ids, ids, test_size=0.05, random_state=42)

# with open('train.txt', mode='wt', encoding='utf-8') as myfile:
#     for s in data_train:
#         myfile.write(s)
#         myfile.write('\n')
# with open('dev.txt', mode='wt', encoding='utf-8') as myfile:
#     for s in data_test:
#         myfile.write(s)
#         myfile.write('\n')