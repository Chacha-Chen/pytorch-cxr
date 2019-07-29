from pathlib import Path
import logging

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
from dataset import STANFORD_CXR_BASE, MIMIC_CXR_BASE, CxrDataset
from utils import logger, print_versions, get_devices, get_ip, get_commit
from adamw import AdamW


class IidSingleTrainEnvironment(PredictEnvironment):

    def __init__(self, device, train_data="stanford", amp_enable=False):
        self.device = device
        self.distributed = False
        self.amp = amp_enable

        stanford_train_set = CxrDataset(STANFORD_CXR_BASE, "train.csv", mode="per_study")
        stanford_test_set = CxrDataset(STANFORD_CXR_BASE, "valid.csv", mode="per_study")
        stanford_concat_set = ConcatDataset([stanford_train_set, stanford_test_set])

        mimic_train_set = CxrDataset(MIMIC_CXR_BASE, "train.csv", mode="per_study")
        mimic_test_set = CxrDataset(MIMIC_CXR_BASE, "valid.csv", mode="per_study")
        mimic_concat_set = ConcatDataset([mimic_train_set, mimic_test_set])

        #datasets = random_split(concat_set, [360000, 15000, len(concat_set) - 375000])
        #subset = Subset(concat_set, range(0, 36))
        #datasets = random_split(subset, [len(subset) - 12, 12])

        num_trainset = 20000
        num_testset = 10000

        self.stanford_datasets = random_split(stanford_concat_set, [num_trainset, num_testset, len(stanford_concat_set) - (num_trainset + num_testset)])
        self.mimic_datasets = random_split(mimic_concat_set, [num_trainset, num_testset, len(mimic_concat_set) - (num_trainset + num_testset)])

        self.train_set = ConcatDataset([self.stanford_datasets[0], self.mimic_datasets[0]])

        pin_memory = True if self.device.type == 'cuda' else False
        self.train_loader = DataLoader(self.train_set, batch_size=32, num_workers=8, shuffle=True, pin_memory=pin_memory)
        self.test_loader = DataLoader(self.stanford_datasets[1], batch_size=32, num_workers=8, shuffle=False, pin_memory=pin_memory)
        self.xtest_loader = DataLoader(self.mimic_datasets[1], batch_size=32, num_workers=8, shuffle=False, pin_memory=pin_memory)

        self.out_dim = len(stanford_train_set.labels)
        self.labels = stanford_train_set.labels

        #img, tar = datasets[0]
        #plt.imshow(img.squeeze(), cmap='gray')

        super().__init__(out_dim=self.out_dim, device=self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')
        self.loss = nn.BCEWithLogitsLoss()

        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def save_model(self, filename):
        filedir = Path(filename).parent.resolve()
        filedir.mkdir(mode=0o755, parents=True, exist_ok=True)
        filepath = Path(filename).resolve()
        logger.debug(f"saving the model to {filepath}")
        state = self.model.state_dict()
        torch.save(state, filename)


class IidDistributedTrainEnvironment(IidSingleTrainEnvironment):

    def __init__(self, devices, local_rank, amp_enable=False):
        self.device = devices[local_rank]
        torch.cuda.set_device(self.device)

        logger.info(f"waiting other ranks ...")
        dist.init_process_group(backend="nccl", init_method="env://")

        self.local_rank = local_rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        logger.info(f"initialized on {self.device} as rank {self.rank} of {self.world_size}")

        super().__init__(self.device, amp_enable=amp_enable)
        self.distributed = True

        pin_memory = True if self.device.type == 'cuda' else False
        self.train_loader = DataLoader(self.train_loader.dataset,
                                       batch_size=self.train_loader.batch_size,
                                       num_workers=self.train_loader.num_workers,
                                       sampler=DistributedSampler(self.train_loader.dataset),
                                       shuffle=False, pin_memory=pin_memory)

        self.model = DistributedDataParallel(self.model, device_ids=[self.device],
                                             output_device=self.device, find_unused_parameters=True)

    def save_model(self, filename):
        # save only if local_rank == 0
        if self.local_rank == 0:
            super().save_model(filename)


class IidTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.progress.update({
            'xtest_accuracy': [],
            'xtest_auc_score': [],
        })

        xtest_percent = len(self.env.xtest_loader.sampler) / len(self.env.xtest_loader.dataset) * 100.
        logger.info(f"using {len(self.env.xtest_loader.sampler)}/{len(self.env.xtest_loader.dataset)} ({xtest_percent:.1f}%) entries for cross testing")

    def train(self, num_epoch, start_epoch=1):
        if start_epoch > 1:
            model_path = runtime_path.joinpath(f"model_epoch_{(start_epoch - 1):03d}.pth.tar")
            self.env.load_model(model_path)
            if self.env.distributed:
                self.env.train_loader.sampler.set_epoch(start_epoch - 1)

        for epoch in range(start_epoch, num_epoch + 1):
            self.train_epoch(epoch)
            self.test(epoch, self.env.test_loader)
            self.test(epoch, self.env.xtest_loader, prefix="xtest_")
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
                self.test(epoch, self.env.xtest_loader, prefix="xtest_")
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
                self.test(epoch, self.env.xtest_loader, prefix="mimic_")
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

    runtime_path, devices = initialize(args)

    # start training
    if args.local_rank is None:
        logger.info(f"using {args.main_dataset} as the main dataset")
        assert args.main_dataset in ["stanford", "mimic"]
        env = IidSingleTrainEnvironment(devices[0], train_data=args.main_dataset, amp_enable=args.amp)
    else:
        env = IidDistributedTrainEnvironment(devices, args.local_rank, amp_enable=args.amp)

    t = IidTrainer(env, runtime_path=runtime_path, tensorboard=args.tensorboard)
    t.train(args.epoch, start_epoch=args.start_epoch)
    #t.cross_test_only(args.epoch, start_epoch=args.start_epoch)
    #t.dist_cross_test_only(args.epoch, start_epoch=args.start_epoch)

