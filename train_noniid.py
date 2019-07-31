from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.utils.data import ConcatDataset, Subset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

import torchvision
import torchnet as tnt

from apex import amp

from predict import PredictEnvironment, Predictor
from train import Trainer, initialize
from dataset import STANFORD_CXR_BASE, MIMIC_CXR_BASE, NIH_CXR_BASE, CxrDataset, CxrConcatDataset, CxrSubset, cxr_random_split
from utils import logger, print_versions, get_devices, get_ip, get_commit
from adamw import AdamW


DATASETS = ["stanford", "mimic", "nih"]


class NoniidSingleTrainEnvironment(PredictEnvironment):

    def __init__(self, device, train_data="stanford", amp_enable=False):
        self.device = device
        self.distributed = False
        self.amp = amp_enable

        self.local_rank = 0
        self.rank = 0

        stanford_train_set = CxrDataset(STANFORD_CXR_BASE, "train.csv", num_labels=14, mode="per_study")
        stanford_test_set = CxrDataset(STANFORD_CXR_BASE, "valid.csv", num_labels=14, mode="per_study")
        stanford_set = CxrConcatDataset([stanford_train_set, stanford_test_set])

        mimic_train_set = CxrDataset(MIMIC_CXR_BASE, "train.csv", num_labels=14, mode="per_study")
        mimic_test_set = CxrDataset(MIMIC_CXR_BASE, "valid.csv", num_labels=14, mode="per_study")
        mimic_set = CxrConcatDataset([mimic_train_set, mimic_test_set])

        nih_set = CxrDataset(NIH_CXR_BASE, "Data_Entry_2017.csv", num_labels=15, mode="per_study")

        set_splits = [20000, 10000]

        self.stanford_datasets = cxr_random_split(stanford_set, set_splits)
        self.mimic_datasets = cxr_random_split(mimic_set, set_splits)
        self.nih_datasets = cxr_random_split(nih_set, set_splits)

        if train_data == "stanford":
            self.set_data_loader(self.stanford_datasets, None, 12, 8)
        elif train_data == "mimic":
            self.set_data_loader(self.mimic_datasets, None, 12, 8)
        else:
            self.set_data_loader(self.nih_datasets, None, 12, 8)

        self.labels = self.train_loader.dataset.labels
        self.out_dim = len(self.labels)
        self.positive_weights = torch.FloatTensor(self.get_positive_weights()).to(device)

        #img, tar = datasets[0]
        #plt.imshow(img.squeeze(), cmap='gray')

        super().__init__(out_dim=self.out_dim, device=self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.positive_weights)

        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def set_data_loader(self, main_datasets, xtest_datasets=None, batch_size=32, num_workers=8):
        pin_memory = True if self.device.type == 'cuda' else False
        self.train_loader = DataLoader(main_datasets[0], batch_size=batch_size, num_workers=num_workers,
                                       shuffle=True, pin_memory=pin_memory)
        self.test_loader = DataLoader(main_datasets[1], batch_size=batch_size * 3, num_workers=num_workers,
                                      shuffle=False, pin_memory=pin_memory)
        if xtest_datasets is not None:
            self.xtest_loaders = [DataLoader(datasets[1], batch_size=batch_size * 3, num_workers=num_workers,
                                             shuffle=False, pin_memory=pin_memory)
                                  for datasets in xtest_datasets]
        else:
            self.xtest_loaders = []

    def get_positive_weights(self):
        df = self.train_loader.dataset.get_label_counts()
        ratio = df.loc[0] / df.loc[1]
        return ratio.values.tolist()

    def save_model(self, filename):
        filedir = Path(filename).parent.resolve()
        filedir.mkdir(mode=0o755, parents=True, exist_ok=True)
        filepath = Path(filename).resolve()
        logger.debug(f"saving the model to {filepath}")
        state = self.model.state_dict()
        torch.save(state, filename)


class NoniidDistributedTrainEnvironment(NoniidSingleTrainEnvironment):

    def __init__(self, device, local_rank, amp_enable=False):
        rank = dist.get_rank()

        super().__init__(device, train_data=DATASETS[rank], amp_enable=amp_enable)
        self.distributed = True
        self.local_rank = local_rank
        self.world_size = dist.get_world_size()
        self.rank = rank
        logger.info(f"initialized on {device} as rank {self.rank} of {self.world_size}")

        if self.rank == 0:
            self.set_data_loader(self.stanford_datasets, None, 32, 8)
        elif self.rank == 1:
            self.set_data_loader(self.mimic_datasets, None, 32, 8)
        else:
            self.set_data_loader(self.nih_datasets, None, 32, 8)

        #self.model = DistributedDataParallel(self.model, device_ids=[self.device],
        #                                     output_device=self.device, find_unused_parameters=True)
        self.model.to_distributed(self.device)


class NoniidTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.progress.update({
            'xtest_accuracy': [],
            'xtest_auc_score': [],
        })

        for i, xtest in enumerate(self.env.xtest_loaders):
            xtest_percent = len(self.env.xtest.sampler) / len(self.env.xtest.dataset) * 100.
            logger.info(f"using {len(self.env.xtest.sampler)}/{len(self.env.xtest.dataset)} ({xtest_percent:.1f}%) entries for cross testing {i}")

    def train(self, num_epoch, start_epoch=1):
        if start_epoch > 1:
            model_path = runtime_path.joinpath(f"model_epoch_{(start_epoch - 1):03d}.{self.env.rank}.pth.tar")
            self.env.load_model(model_path)
            self.load()

        for epoch in range(start_epoch, num_epoch + 1):
            self.train_epoch(epoch)
            self.test(epoch, self.env.test_loader)
            for i, xtest in enumerate(self.env.xtest_loaders):
                self.test(epoch, xtest, prefix=f"xtest{i}_")
            self.save()

    def cross_test_only(self, num_epoch, start_epoch=1):
        self.load()
        self.progress.update({
            'xtest_accuracy': [],
            'xtest_auc_score': [],
        })

        for epoch in range(start_epoch, num_epoch + 1):
            model_path = runtime_path.joinpath(f"model_epoch_{epoch:03d}.pth.tar")
            try:
                self.env.load_model(model_path)
                for i, xtest in enumerate(self.env.xtest_loaders):
                    self.test(epoch, xtest, prefix=f"xtest{i}_")
                self.save()
            except:
                break

    def dist_cross_test_only(self, num_epoch, start_epoch=1):
        import pickle
        filepath = self.runtime_path.joinpath(f"train.0.pkl")
        with open(filepath, 'rb') as f:
            self.progress = pickle.load(f)
        self.progress.update({
            'stanford_accuracy': [],
            'stanford_auc_score': [],
            'mimic_accuracy': [],
            'mimic_auc_score': [],
        })

        for epoch in range(start_epoch, num_epoch + 1):
            model_path = runtime_path.joinpath(f"model_epoch_{epoch:03d}.pth.tar")
            try:
                self.env.load_model(model_path)
                self.test(epoch, self.env.test_loader, prefix="stanford_")
                #self.test(epoch, self.env.xtest_loader, prefix="mimic_")
                self.save()
            except:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CXR Training")
    # for training
    parser.add_argument('--cuda', default=None, type=str, help="use GPUs with its device ids, separated by commas")
    parser.add_argument('--amp', default=False, action='store_true', help="use automatic mixed precision for faster training")
    parser.add_argument('--epoch', default=100, type=int, help="max number of epochs")
    parser.add_argument('--start-epoch', default=1, type=int, help="start epoch, especially need to continue from a stored model")
    parser.add_argument('--runtime-dir', default='./runtime', type=str, help="runtime directory to store log, pretrained models, and tensorboard metadata")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="true if logging to tensorboard")
    parser.add_argument('--slack', default=False, action='store_true', help="true if logging to slack")
    parser.add_argument('--main-dataset', default='stanford', type=str, help="main dataset for training (single mode only)")
    parser.add_argument('--local_rank', default=None, type=int, help="this is for the use of torch.distributed.launch utility")
    args = parser.parse_args()

    distributed, runtime_path, device = initialize(args)

    # start training
    if distributed:
        env = NoniidDistributedTrainEnvironment(device, args.local_rank, amp_enable=args.amp)
    else:
        logger.info(f"using {args.main_dataset} as the main dataset")
        assert args.main_dataset in DATASETS
        env = NoniidSingleTrainEnvironment(device, train_data=args.main_dataset, amp_enable=args.amp)

    t = NoniidTrainer(env, runtime_path=runtime_path, tensorboard=args.tensorboard)
    t.train(args.epoch, start_epoch=args.start_epoch)
    #t.cross_test_only(args.epoch, start_epoch=args.start_epoch)
    #t.dist_cross_test_only(args.epoch, start_epoch=args.start_epoch)
