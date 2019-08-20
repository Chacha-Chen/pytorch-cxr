import sys
import bisect
from pathlib import Path

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

MODES = ["per_image", "per_study"]
MIN = 320
MAX_CHS = 20


def _load_manifest(file_path, num_labels=14, mode="per_study"):
    assert mode in MODES
    if not file_path.exists():
        logger.error(f"manifest file {file_path} not found.")
        sys.exit(1)

    logger.debug(f"loading dataset manifest {file_path} ...")
    df = pd.read_csv(str(file_path)).fillna(0)
    #if mode == "per_image":
    #    df = df[(df['Frontal/Lateral'] == 'Frontal') & (df['AP/PA'] == 'PA')]
    #LABELS = df.columns[-num_labels:].values.tolist()
    LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    #if LABELS[0] != "No Finding":
    #    idx = LABELS.index("No Finding")
    #    LABELS[0], LABELS[idx] = LABELS[idx], LABELS[0]
    paths = df[df.columns[0]]
    orients = df['Frontal/Lateral'].replace({'Frontal': '0', 'Lateral': '1'})
    labels = df[LABELS].astype(int).replace(-1, 1)  # substitute uncertainty to positive
    df_tmp = pd.concat([paths, orients, labels], axis=1)
    if mode == "per_image":
        entries = df_tmp
    elif mode == "per_study":
        logger.debug("grouping by studies ... ")
        df_tmp['study'] = df_tmp.apply(lambda x: str(Path(x[0]).parent), axis=1)
        df_tmp.set_index(['study'], inplace=True)
        aggs = { df_tmp.columns[0]: lambda x: ','.join(x.astype(str)) }
        aggs.update({ df_tmp.columns[1]: lambda x: ','.join(x.astype(str)) })
        aggs.update({ x: 'mean' for x in LABELS })
        df_tmp = df_tmp.groupby(['study']).agg(aggs).reset_index(0, drop=True)
        entries = df_tmp
    else:
        raise RuntimeError

    logger.debug(f"{len(entries)} entries are loaded.")
    return entries


MEAN = 0.4
STDEV = 0.2

cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    #tfms.ColorJitter(),
    tfms.Resize(MIN+10, Image.LANCZOS),
    #tfms.RandomRotation((-10, 10)),
    tfms.RandomCrop((MIN, MIN)),
    #tfms.RandomHorizontalFlip(),
    #tfms.RandomVerticalFlip(),
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


def get_image(img_path, transforms, use_memcache=True):
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
def get_study(img_paths, orients, transforms, use_memcache=True):
    image_tensor = torch.randn(MAX_CHS, MIN, MIN) * STDEV + MEAN
    for i, img_path in enumerate(img_paths):
        if use_memcache:
            image = fetch_image(img_path)
        else:
            image = imageio.imread(img_path, as_gray=True)
        imgs = transforms(image)
        image_tensor[i, :, :] = transforms(image)
    if transforms == cxr_train_transforms:
        image_tensor = image_tensor[torch.randperm(MAX_CHS), :, :]
    return image_tensor


class CxrDataset(Dataset):

    transforms = cxr_train_transforms

    def __init__(self, base_path, manifest_file, num_labels=14, mode="per_study", *args, **kwargs):
        super().__init__(*args, **kwargs)
        manifest_path = base_path.joinpath(manifest_file).resolve()
        self.entries = _load_manifest(manifest_path, num_labels, mode)
        self.base_path = base_path
        self.mode = mode

    def __getitem__(self, index):

        def get_entries(index):
            df = self.entries.iloc[index]
            paths = [self.base_path.joinpath(x).resolve() for x in df[0].split(',')]
            orients = [int(x) for x in df[1].split(',')]
            label = df[2:].tolist()
            return paths, orients, label

        if self.mode == "per_image":
            img_paths, orients, label = get_entries(index)
            image_tensor = get_image(img_paths[0], CxrDataset.transforms)
            target_tensor = torch.FloatTensor(label)
        elif self.mode == "per_study":
            img_paths, orients, label = get_entries(index)
            image_tensor = get_study(img_paths, orients, CxrDataset.transforms)
            target_tensor = torch.FloatTensor(label)
        else:
            raise RuntimeError

        return image_tensor, target_tensor

    def __len__(self):
        return len(self.entries)

    def get_label_counts(self, indices=None):
        df = self.entries if indices is None else self.entries.iloc[indices]
        counts = [df[x].value_counts() for x in self.labels]
        new_df = pd.concat(counts, axis=1).fillna(0).astype(int)
        return new_df

    @property
    def labels(self):
        return self.entries.columns[2:].values.tolist()

    @staticmethod
    def train():
        CxrDataset.transforms = cxr_train_transforms

    @staticmethod
    def eval():
        CxrDataset.transforms = cxr_test_transforms


class CxrConcatDataset(ConcatDataset):

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.get_label_counts()

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
            assert len(df.columns) == len(dataset.labels), "label names should be matched!"
        return df

    @property
    def labels(self):
        return self.datasets[0].labels


class CxrSubset(Subset):

    #def __init__(self, *args, **kwargs):
    #    super().__init__(*args, **kwargs)
    #    self.get_label_counts()

    def get_label_counts(self, indices=None):
        if indices is None:
            indices = list(range(self.__len__()))
        df = self.dataset.get_label_counts([self.indices[x] for x in indices])
        return df

    @property
    def labels(self):
        return self.dataset.labels


def cxr_random_split(dataset, lengths):
    from torch._utils import _accumulate
    if sum(lengths) > len(dataset):
        raise ValueError("Sum of input lengths must less or equal to the length of the input dataset!")
    indices = torch.randperm(sum(lengths)).tolist()
    return [CxrSubset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


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
