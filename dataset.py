import sys
import bisect
from pathlib import Path
from collections import Counter
from enum import Enum

import numpy as np
import imageio
import pandas as pd
from tqdm import tqdm
from PIL import Image
import bmemcached as bmc

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as tfms

from utils import logger


pd.set_option('mode.chained_assignment', None)


#CXR_BASE = Path("/mnt/hdd/cxr").resolve()
CXR_BASE = Path("./data").resolve()
STANFORD_CXR_BASE = CXR_BASE.joinpath("stanford/v1").resolve()
MIMIC_CXR_BASE = CXR_BASE.joinpath("mimic/v1").resolve()
NIH_CXR_BASE = CXR_BASE.joinpath("nih/v1").resolve()

MIN = 256
MAX_CHS = 11
MEAN = 0.4
STDEV = 0.2

"""
cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.ColorJitter(),
    tfms.Resize(MIN+10, Image.LANCZOS),
    #tfms.RandomRotation((-10, 10)),
    tfms.RandomCrop((MIN, MIN)),
    #tfms.RandomHorizontalFlip(),
    #tfms.RandomVerticalFlip(),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
    #tfms.Normalize((0.1307,), (0.3081,))
])
"""
cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomAffine((-5, 5), translate=None, scale=None, shear=(0.9, 1.1)),
    #tfms.RandomRotation((-5, 5), resample=Image.BICUBIC),
    #tfms.Resize(MIN+10, Image.LANCZOS),
    #tfms.RandomCrop((MIN, MIN)),
    #tfms.RandomHorizontalFlip(),
    #tfms.RandomVerticalFlip(),
    tfms.RandomResizedCrop((MIN, MIN), scale=(0.5, 0.75), ratio=(0.95, 1.05), interpolation=Image.LANCZOS),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
    #tfms.Normalize((0.1307,), (0.3081,))
])


cxr_test_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize(MIN, Image.LANCZOS),
    tfms.CenterCrop(MIN),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
    #tfms.Normalize((0.1307,), (0.3081,))
])


client = bmc.Client(('localhost:11211', ))

def fetch_image(img_path):
    image = client.get(str(img_path))
    if image is None:
        image = imageio.imread(img_path, as_gray=True)
        client.set(str(img_path), image)
    return image


def get_image(img_path, transforms, use_memcache=False):
    if use_memcache:
        image = fetch_image(img_path)
    else:
        image = imageio.imread(img_path, as_gray=True)
    image_tensor = transforms(image)
    return image_tensor

"""
def get_study(img_paths, orients, transforms):
    restruct = [[], []]
    for img_path, orient in zip(img_paths, orients):
        restruct[orient].append(img_path)

    def make_group(max_chs, img_paths):
        image_tensor = torch.zeros(max_chs, MIN, MIN)
        for i, img_path in enumerate(img_paths):
            image = fetch_image(img_path)
            image_tensor[i, :, :] = transforms(image)
        if transforms == cxr_train_transforms:
            image_tensor = image_tensor[torch.randperm(max_chs), :, :]
        return image_tensor

    tensors = [make_group(int(MAX_CHS / 2), x) for x in restruct]
    image_tensor = torch.cat(tensors, dim=0)
    return image_tensor
"""
def get_study(img_paths, transforms, use_memcache=False):
    image_tensor = torch.randn(MAX_CHS, MIN, MIN) * STDEV + MEAN
    rand = transforms == cxr_train_transforms
    rand_idx = torch.randperm(len(img_paths))
    for i, img_path in enumerate(img_paths):
        if use_memcache:
            image = fetch_image(img_path)
        else:
            image = imageio.imread(img_path, as_gray=True)
        j = rand_idx[i] if rand else i
        image_tensor[j, :, :] = transforms(image)
    return image_tensor


class Mode(Enum):
    PER_IMAGE = 0
    PER_STUDY = 1


class CxrDataset(Dataset):

    def __init__(self, base_path, manifest_file, mode=Mode.PER_IMAGE, classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = base_path
        self.mode = mode
        self.classes = classes
        self.transforms = cxr_train_transforms
        self.prepare_entries(manifest_file)

    def __getitem__(self, index):

        def get_entries(index):
            df = self.entries.iloc[index]
            paths = [self.base_path.joinpath(x).resolve() for x in df[0].split(',')]
            label = df[1:].tolist()
            return paths, label

        if self.mode == Mode.PER_IMAGE:
            img_paths, label = get_entries(index)
            image_tensor = get_image(img_paths[0], self.transforms)
            target_tensor = torch.FloatTensor(label)
            channels = 1
        else:  # Mode.PER_STUDY
            img_paths, label = get_entries(index)
            image_tensor = get_study(img_paths, self.transforms)
            target_tensor = torch.FloatTensor(label)
            channels = len(img_paths)

        return image_tensor, target_tensor, channels

    def __len__(self):
        return len(self.entries)

    def load_manifest(self, file_path):
        raise NotImplementedError

    def prepare_entries(self, manifest_file):
        manifest_path = self.base_path.joinpath(manifest_file).resolve()
        if not manifest_path.exists():
            logger.error(f"manifest file {file_path} not found.")
            sys.exit(1)
        df_tmp = self.load_manifest(manifest_path)

        if self.mode == Mode.PER_IMAGE:
            self.entries = df_tmp
        else:  # Mode.PER_STUDY
            logger.debug("grouping by studies ... ")
            df_tmp['study'] = df_tmp.apply(lambda x: str(Path(x[0]).parent), axis=1)
            df_tmp.set_index(['study'], inplace=True)
            aggs = { df_tmp.columns[0]: lambda x: ','.join(x.astype(str)) }
            aggs.update({ x: 'mean' for x in self.classes })
            df_tmp = df_tmp.groupby(['study']).agg(aggs).reset_index(0, drop=True)
            self.entries = df_tmp

        logger.debug(f"{len(self.entries)} entries are loaded.")

    def get_label_counts(self, indices=None):
        df = self.entries if indices is None else self.entries.iloc[indices]
        counts = [df[x].value_counts() for x in self.classes]
        new_df = pd.concat(counts, axis=1).fillna(0).astype(int)
        return new_df

    def rename_classes(self, class_dict):
        for k, v in class_dict.items():
            idx = self.classes.index(k)
            self.classes[idx] = v
        self.entries.rename(columns=class_dict, inplace=True)

    def train(self):
        self.transforms = cxr_train_transforms

    def eval(self):
        self.transforms = cxr_test_transforms


class StanfordCxrDataset(CxrDataset):

    NUM_CLASSES = 14

    def __init__(self, manifest_file, *args, **kwargs):
        super().__init__(STANFORD_CXR_BASE, manifest_file, *args, **kwargs)

    def load_manifest(self, file_path):
        logger.debug(f"loading dataset manifest {file_path} ...")
        df = pd.read_csv(str(file_path)).fillna(0)
        #df = df.loc[df['AP/PA'] == 'PA']
        if self.classes is None:
            self.classes = df.columns[-StanfordCxrDataset.NUM_CLASSES:].values.tolist()
        #if self.classes[0] != "No Finding":
        #    idx = self.classes.index("No Finding")
        #    self.classes[0], self.classes[idx] = self.classes[idx], self.classes[0]
        paths = df[df.columns[0]]
        labels = df[self.classes].astype(int).replace(-1, 1)  # substitute uncertainty to positive
        return pd.concat([paths, labels], axis=1)


class MitCxrDataset(CxrDataset):

    NUM_CLASSES = 14

    def __init__(self, manifest_file, *args, **kwargs):
        super().__init__(MIMIC_CXR_BASE, manifest_file, *args, **kwargs)

    def load_manifest(self, file_path):
        logger.debug(f"loading dataset manifest {file_path} ...")
        df = pd.read_csv(str(file_path)).fillna(0)
        #df = df.loc[df['AP/PA'] == 'PA']
        if self.classes is None:
            self.classes = df.columns[-MitCxrDataset.NUM_CLASSES:].values.tolist()
        #if self.classes[0] != "No Finding":
        #    idx = self.classes.index("No Finding")
        #    self.classes[0], self.classes[idx] = self.classes[idx], self.classes[0]
        paths = df[df.columns[0]]
        labels = df[self.classes].astype(int).replace(-1, 1)  # substitute uncertainty to positive
        return pd.concat([paths, labels], axis=1)


class NihCxrDataset(CxrDataset):

    NUM_CLASSES = 15

    def __init__(self, manifest_file, *args, **kwargs):
        super().__init__(NIH_CXR_BASE, manifest_file, *args, **kwargs)

    def load_manifest(self, file_path):
        logger.debug(f"loading dataset manifest {file_path} ...")
        df = pd.read_csv(str(file_path)).fillna(0)
        #df = df.loc[df['AP/PA'] == 'PA']
        if self.classes is None:
            self.classes = df.columns[-NihCxrDataset.NUM_CLASSES:].values.tolist()
        paths = df[df.columns[0]]
        labels = df[self.classes].astype(int).replace(-1, 1)  # substitute uncertainty to positive
        return pd.concat([paths, labels], axis=1)


class CxrConcatDataset(ConcatDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.get_label_counts()
        self.check_classes()

    def check_classes(self):
        tmp = Counter()
        for dataset in self.datasets:
            tmp.update(dataset.classes)
        for k, v in tmp.items():
            assert v == len(self.datasets), "class names should be matched!"

    def get_label_counts(self, indices=None):
        if indices is None:
            indices = list(range(self.__len__()))
        dataset_indices = [bisect.bisect_right(self.cumulative_sizes, idx) for idx in indices]
        sample_indices = [(i if d == 0 else i - self.cumulative_sizes[d - 1]) for i, d in zip(indices, dataset_indices)]
        nested_indices = [[] for d in self.datasets]
        for d, s in zip(dataset_indices, sample_indices):
            nested_indices[d].append(s)
        dfs = []
        for d, dataset in enumerate(self.datasets):
            dfs.append(dataset.get_label_counts(nested_indices[d]))
        df = pd.concat(dfs, sort=False).groupby(level=0).sum().astype(int)
        for dataset in self.datasets:
            assert len(df.columns) == len(dataset.classes), "class names should be matched!"
        return df

    def rename_classes(self, class_dict):
        for dataset in self.datasets:
            dataset.rename_classes(class_dict)

    def train(self):
        for dataset in self.datasets:
            dataset.train()

    def eval(self):
        for dataset in self.datasets:
            dataset.eval()

    @property
    def classes(self):
        return self.datasets[0].classes


class CxrSubset(Subset):

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.get_label_counts()

    def get_label_counts(self, indices=None):
        if indices is None:
            indices = list(range(self.__len__()))
        df = self.dataset.get_label_counts([self.indices[x] for x in indices])
        return df

    def rename_classes(self, class_dict):
        self.dataset.rename_classes(class_dict)

    def train(self):
        self.dataset.train()

    def eval(self):
        self.dataset.eval()

    @property
    def classes(self):
        return self.dataset.classes


def cxr_random_split(dataset, lengths):
    if sum(lengths) > len(dataset):
        raise ValueError("Sum of input lengths must less or equal to the length of the input dataset!")
    indices = torch.randperm(sum(lengths)).split(lengths)
    return [CxrSubset(dataset, idx.tolist()) for idx in indices]


MIN_RES = 512

def copy_stanford_dataset(src_path, image_process=True):
    for m in [src_path.joinpath("train.csv"), src_path.joinpath("valid.csv")]:
        print(f">>> processing {m}...")
        df = pd.read_csv(str(m))
        for i in tqdm(range(len(df)), total=len(df), dynamic_ncols=True):
            f = df.iloc[i]["Path"].split('/', 1)[1]
            ff = src_path.joinpath(f).resolve()
            if image_process:
                img = Image.open(ff)
                w, h = img.size
                rs = (MIN_RES, int(h/w*MIN_RES)) if w < h else (int(w/h*MIN_RES), MIN_RES)
                resized = img.resize(rs, Image.LANCZOS)
            r = ff.relative_to(src_path)
            t = STANFORD_CXR_BASE.joinpath(r).resolve()
            #print(f"{ff} -> {t}")
            if image_process:
                Path.mkdir(t.parent, parents=True, exist_ok=True)
                resized.save(t, "JPEG")
            df.at[i, "Path"] = f
        r = m.relative_to(src_path).name
        t = STANFORD_CXR_BASE.joinpath(r).resolve()
        df.to_csv(t, float_format="%.0f", index=False)


def copy_mimic_dataset(src_path, image_process=True):
    for m in [src_path.joinpath("train.csv"), src_path.joinpath("valid.csv")]:
        print(f">>> processing {m}...")
        df = pd.read_csv(str(m))
        for i in tqdm(range(len(df)), total=len(df), dynamic_ncols=True):
            f = df.iloc[i]["path"]
            ff = src_path.joinpath(f).resolve()
            if image_process:
                img = Image.open(ff)
                w, h = img.size
                rs = (MIN_RES, int(h/w*MIN_RES)) if w < h else (int(w/h*MIN_RES), MIN_RES)
                resized = img.resize(rs, Image.LANCZOS)
            r = ff.relative_to(src_path)
            t = MIMIC_CXR_BASE.joinpath(r).resolve()
            #print(f"{ff} -> {t}")
            if image_process:
                Path.mkdir(t.parent, parents=True, exist_ok=True)
                resized.save(t, "JPEG")
        df.rename(columns={"Airspace Opacity": "Lung Opacity"}) # to match stanford's label
        r = m.relative_to(src_path).name
        t = MIMIC_CXR_BASE.joinpath(r).resolve()
        df.to_csv(t, float_format="%.0f", index=False)


def copy_nih_dataset(src_path, image_process=True):
    manifest_file = src_path.joinpath("Data_Entry_2017.csv")
    print(f">>> processing {manifest_file}...")
    df = pd.read_csv(str(manifest_file))
    files_list = {}
    for f in src_path.rglob("*.png"):
        files_list[f.name] = f
    df_tmps = []
    for i, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True):
        f = row["Image Index"]
        patient = row["Patient ID"]
        study = row["Follow-up #"]
        ff = files_list[f]
        if image_process:
            img = Image.open(ff)
            w, h = img.size
            rs = (MIN_RES, int(h/w*MIN_RES)) if w < h else (int(w/h*MIN_RES), MIN_RES)
            resized = img.resize(rs, Image.LANCZOS).convert('L')
        r = ff.relative_to(src_path)
        t = NIH_CXR_BASE.joinpath(r).resolve()
        basename, filename = t.parent, t.name
        t = basename.joinpath(f"patient{patient:05d}", f"study{study:03d}", filename)
        t = Path(str(t).replace('.png', '.jpg'))
        #print(f"{ff} -> {t}")
        if image_process:
            Path.mkdir(t.parent, parents=True, exist_ok=True)
            resized.save(t, "JPEG")
        df_tmp = pd.DataFrame()
        df_tmp["path"] = [t.relative_to(NIH_CXR_BASE)]
        for l in row["Finding Labels"].split('|'):
            df_tmp[l] = [1]
        df_tmps.append(df_tmp)
    df2 = pd.concat(df_tmps, sort=False)
    r = manifest_file.name
    t = NIH_CXR_BASE.joinpath(r).resolve()
    df2.to_csv(t, float_format="%.0f", index=False)


if __name__ == "__main__":
    """
    manifest_file = "CheXpert-v1.0-small/train.csv"
    file_path = STANFORD_CXR_BASE.joinpath(manifest_file).resolve()
    entries, labels = _load_manifest(file_path, mode="per_study")
    #from pprint import pprint
    #pprint(entries[:20])
    """

    # resize & copy stanford dataset
    src_path = Path("/media/nfs/CXR/Stanford/full_resolution_version/CheXpert-v1.0").resolve()
    if src_path.exists():
        copy_stanford_dataset(src_path)

    # resize & copy mimic dataset
    src_path = Path("/media/mycloud/MIMIC_CXR").resolve()
    if src_path.exists():
        copy_mimic_dataset(src_path)

    # resize & copy nih dataset
    src_path = Path("/mnt/hdd/cxr/nih/original").resolve()
    if src_path.exists():
        copy_nih_dataset(src_path)
