import torch
from torchvision import transforms
from mil.io.reader import read_record, peek
import os
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, df_meta, gender_list, mutant_list, genomic ,transform=True):
        if transform:
            self.transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((0,180)),
                    # for more transforms, ask professor
                ]
            )
        self.transform = transform
        self.tau = 365.25 * 2
        self.path_list = [
            os.path.join(data_path, folder, tfr)
            for folder, tfr in zip(df_meta["set"], df_meta[".tfr"])
        ]
        self.genders = dict(
            [
                (os.path.join(data_path, folder, tfr), gend)
                for folder, tfr, gend in zip(
                    df_meta["set"], df_meta[".tfr"], df_meta["Gender"]
                )
            ]
        )
        self.gender_lookup = dict([(gend, i) for i, gend in enumerate(gender_list)])
        self.mutants = dict(
            [
                (os.path.join(data_path, folder, tfr), mut)
                for folder, tfr, mut in zip(
                    df_meta["set"], df_meta[".tfr"], df_meta["Molecular subtype"]
                )
            ]
        )
        self.mutant_lookup = dict([(mut, i) for i, mut in enumerate(mutant_list)])
        self.ages = dict(
            [
                (os.path.join(data_path, folder, tfr), ag)
                for folder, tfr, ag in zip(
                    df_meta["set"], df_meta[".tfr"], df_meta["Age at diagnosis"]
                )
            ]
        )
        self.labels = dict(
            [
                (os.path.join(data_path, folder, tfr), label)
                for folder, tfr, label in zip(
                    df_meta["set"], df_meta[".tfr"], df_meta["Event < 2 years"]
                )
            ]
        )
        # self.label_lookup = dict([(labl, i) for i, labl in enumerate(label_list)])
        self.genomic = genomic
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        # serialized = list(tf.data.TFRecordDataset(self.path_list[index]))[0]
        # variables = peek(serialized)
        # features, labels, slide_meta, tile_meta = read_record(serialized, variables, structured=False)
        # features = torch.tensor(np.array(features))
        # y = tf.cast(labels['time'][0], tf.float32)>=365.25 * 2
        # y = int(y.numpy())
        features = np.load(self.path_list[index].replace('tfr','npz'))
        features = torch.tensor(features['arr_0'])
        x,y,z = features.shape
        features = features.view(z,x,y)
        if self.transform:
            features = self.transforms(features)
        label = self.labels[self.path_list[index]]*1.0
        if self.genomic:
            gender = self.gender_lookup[self.genders[self.path_list[index]]]
            age = self.ages[self.path_list[index]]
            mutant = self.mutant_lookup[self.mutants[self.path_list[index]]]
        else:
            gender = -1.0
            age = -1.0
            mutant = -1.0
        return self.path_list[index],features, gender, age, mutant, label
