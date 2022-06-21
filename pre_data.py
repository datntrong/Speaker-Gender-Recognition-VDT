import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image
import librosa
import torch
import config as cf
import wavencoder
import torchaudio
from sklearn.model_selection import train_test_split

def extract_spec(wav):
    linear = librosa.stft(wav, n_fft=2048, hop_length=512)
    features, _ = librosa.magphase(linear)
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)

    spectrogram = np.array(Image.fromarray(features).resize(size=(224, 224)))

    out = np.zeros((3, 224, 224), dtype=np.float32)
    out[0, :, :] = spectrogram
    out[1, :, :] = spectrogram
    out[2, :, :] = spectrogram
    return torch.tensor(out)


def extract_melspec(wav):
    features = librosa.feature.melspectrogram(y=wav, sr=8000, n_fft=512, hop_length=160, win_length=400)
    features = librosa.power_to_db(features, ref=1.0, amin=1e-10, top_db=None)
    return features


class MyDatasetSTFT(Dataset):
    def __init__(self, filenames, labels, transform=None, duration=2, data_type='spectral'):
        assert len(filenames) == len(labels), "Number of files != number of labels"
        self.fns = filenames
        self.lbs = labels
        self.transform = transform
        self.duration = duration
        self.noise_dataset_path = 'noise_dataset'
        self.get_transforms = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=duration * 8000, pad_position='center',
                                          crop_position='random'),
        ])

        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=40, sample_rate=8000, log_mels=True)
        self.data_type = data_type

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        fname = self.fns[idx]
        wav, samplerate = torchaudio.load(fname)
        transformed_audio = self.get_transforms(wav)
        if self.data_type =='spectral':
            transformed_audio = transformed_audio[0].cpu().numpy()
            feats = extract_spec(transformed_audio)

        elif self.data_type == 'mfcc':
            feats = self.mfcc_transform(transformed_audio)
        elif self.data_type == 'melspec':
            transformed_audio = transformed_audio[0].cpu().numpy()
            feats = extract_melspec(transformed_audio)
        elif self.data_type == 'tdnn_mfcc':
            feats = self.mfcc_transform(transformed_audio)
            feats = feats.T
        return torch.tensor(feats), self.lbs[idx], self.fns[idx]


def get_fn_lbs():
    lbs = []
    fns = []
    count = 0
    for name_dir in os.listdir(cf.BASE_TRAIN):
        for i in os.listdir(os.path.join(cf.BASE_TRAIN, name_dir)):
            for j in os.listdir(os.path.join(cf.BASE_TRAIN, name_dir, i)):
                dur = librosa.get_duration(filename=os.path.join(cf.BASE_TRAIN, name_dir, i, j));
                if dur > 1:
                    lbs.append(count)
                    fns.append(os.path.join(cf.BASE_TRAIN, name_dir, i, j))
            print(name_dir, i, end='')
            print(count)
            count += 1

    return lbs, fns


def get_fn_test():
    lbs = []
    fns = []
    df = pd.read_csv(cf.PATH_LABEL_PUBLIC_TEST, header=None,
                     names=["gender", "filename"])
    for index, row in df.iterrows():
        if str(row['gender']) == 'M':
            du = librosa.get_duration(filename=os.path.join(cf.PATH_WAV_PUBLIC_TEST, str(row['filename'])))
            if du > 1:
                lbs.append(1)
                fns.append(os.path.join(cf.PATH_WAV_PUBLIC_TEST, str(row['filename'])))
        elif str(row['gender']) == 'F':
            du = librosa.get_duration(filename=os.path.join(cf.PATH_WAV_PUBLIC_TEST, str(row['filename'])))
            if du > 1:
                lbs.append(0)
                fns.append(os.path.join(cf.PATH_WAV_PUBLIC_TEST, str(row['filename'])))
    return lbs, fns


def get_fn_submit():
    fns = []
    lbs = []
    for i in os.listdir(cf.PATH_WAV_PUBLIC_TEST):
        fns.append(os.path.join(cf.PATH_WAV_PUBLIC_TEST, i))
        lbs.append(0)
    return lbs, fns


def build_dataloaders(args):

    # train
    submit_lbs, submit_fns = get_fn_submit()

    lbs, fns = get_fn_lbs()

    train_fns, val_fns, train_lbs, val_lbs = train_test_split(fns, lbs, test_size=0.1, random_state=42, shuffle=True)
    test_lbs, test_fns = get_fn_test()
    # train_fns, test_fns, train_lbs, test_lbs = train_test_split(train_fns, train_lbs, test_size=0.1, random_state=42,shuffle=True)
    print('First val fn: {}'.format(val_fns[0]))
    print(Counter(train_lbs))
    print(Counter(val_lbs))

    num_classes = len(set(train_lbs))
    print('Total training files: {}'.format(len(train_fns)))
    print('Total validation files: {}'.format(len(val_fns)))
    print('Total classes: {}'.format(num_classes))

    dsets = dict()
    dsets['train'] = MyDatasetSTFT(train_fns, train_lbs, duration=args.duration)
    dsets['val'] = MyDatasetSTFT(val_fns, val_lbs, duration=args.duration)
    dsets['test'] = MyDatasetSTFT(test_fns, test_lbs, duration=args.duration)
    dsets['submit'] = MyDatasetSTFT(submit_fns, submit_lbs, duration=args.duration)

    dset_loaders = dict()
    dset_loaders['train'] = DataLoader(dsets['train'],
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       # sampler = WeightedRandomSampler(durations, args.batch_size, replacement = False),
                                       # sampler = StratifiedSampler_weighted_duration(train_fns, gamma = args.gamma),
                                       # sampler = StratifiedSampler_weighted(train_lbs, batch_size = args.batch_size, gamma = args.gamma),
                                       num_workers=cf.NUM_WORKERS)

    dset_loaders['val'] = DataLoader(dsets['val'],
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=cf.NUM_WORKERS)

    dset_loaders['test'] = DataLoader(dsets['test'],
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=cf.NUM_WORKERS)

    dset_loaders['submit'] = DataLoader(dsets['submit'],
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=cf.NUM_WORKERS)

    return dset_loaders, (train_fns, test_fns, val_fns, train_lbs, test_lbs, val_lbs, submit_lbs, submit_fns)
