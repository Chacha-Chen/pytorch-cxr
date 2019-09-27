from pathlib import Path
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

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

from utils import logger, print_versions, get_devices, get_ip, get_commit
from predict import PredictEnvironment, Predictor
from train import Trainer, initialize
from dataset import StanfordCxrDataset, MitCxrDataset, NihCxrDataset, CxrConcatDataset, CxrSubset, cxr_random_split


DATASETS = ["stanford", "mimic", "nih"]


class NoniidSingleTrainEnvironment(PredictEnvironment):

    def __init__(self, device, train_data="stanford", amp_enable=False):
        self.device = device
        self.distributed = False
        self.amp = amp_enable

        self.train_data = train_data
        self.local_rank = 0
        self.rank = 0

        self.mode = "per_study"

        CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
        stanford_train_set = StanfordCxrDataset("train.csv", mode=self.mode, classes=CLASSES)
        stanford_test_set = StanfordCxrDataset("valid.csv", mode=self.mode, classes=CLASSES)
        stanford_set = CxrConcatDataset([stanford_train_set, stanford_test_set])

        mimic_train_set = MitCxrDataset("train.csv", mode=self.mode, classes=CLASSES)
        mimic_test_set = MitCxrDataset("valid.csv", mode=self.mode, classes=CLASSES)
        mimic_set = CxrConcatDataset([mimic_train_set, mimic_test_set])

        CLASSES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion']
        nih_set = NihCxrDataset("Data_Entry_2017.csv", mode=self.mode, classes=CLASSES)
        nih_set.rename_classes({'Effusion': 'Pleural Effusion'})

        if self.mode == "per_study":
            self.stanford_datasets = cxr_random_split(stanford_set, [175000, 10000])
            self.mimic_datasets = cxr_random_split(mimic_set, [200000, 10000])
            self.nih_datasets = cxr_random_split(nih_set, [100000, 10000])
        else:
            self.stanford_datasets = cxr_random_split(stanford_set, [204000, 10000])
            self.mimic_datasets = cxr_random_split(mimic_set, [357000, 10000])
            self.nih_datasets = cxr_random_split(nih_set, [102000, 10000])

        if train_data == "stanford":
            train_sets = [self.stanford_datasets[0]]
            if self.mode == "per_study":
                batch_size = 7
            else:
                batch_size = 12
        elif train_data == "mimic":
            train_sets = [self.mimic_datasets[0]]
            if self.mode == "per_study":
                batch_size = 8
            else:
                batch_size = 21
        else:
            train_sets = [self.nih_datasets[0]]
            if self.mode == "per_study":
                batch_size = 4
            else:
                batch_size = 6

        test_sets = [self.stanford_datasets[1], self.mimic_datasets[1], self.nih_datasets[1]]
        self.set_data_loader(train_sets, test_sets, batch_size=batch_size)

        self.classes = [x.lower() for x in self.train_loader.dataset.classes]
        self.out_dim = len(self.classes)

        super().__init__(out_dim=self.out_dim, device=self.device, mode=self.mode)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        #self.optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

        self.scheduler = None
        #self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 0.5 ** epoch)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')

        self.positive_weights = torch.FloatTensor(self.get_positive_weights()).to(device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.positive_weights, reduction='none')
        #self.loss = nn.BCEWithLogitsLoss(reduction='none')

        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def set_data_loader(self, train_sets, test_sets, batch_size=6, num_workers=4):
        #num_trainset = 20000
        train_group_id = int(self.rank / len(DATASETS))
        logger.info(f"rank {self.rank} sets {self.train_data} group {train_group_id}")
        #train_set = CxrSubset(train_sets[train_group_id], list(range(num_trainset)))
        train_set = train_sets[train_group_id]
        #test_group_id = train_group_id + 1
        #test_set = test_sets[test_group_id]

        pin_memory = True if self.device.type == 'cuda' else False
        self.train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                                       shuffle=True, pin_memory=pin_memory)
        self.test_loaders = [DataLoader(test_set, batch_size=batch_size * 2, num_workers=num_workers,
                                        shuffle=False, pin_memory=pin_memory)
                             for test_set in test_sets]

    def get_positive_weights(self):
        train_df = self.train_loader.dataset.get_label_counts()
        logger.info(f"train label counts\n{train_df}")
        for i, test_loader in enumerate(self.test_loaders):
            test_df = test_loader.dataset.get_label_counts()
            logger.info(f"test{i} label counts\n{test_df}")
        ratio = train_df.loc[0] / train_df.loc[1]
        return ratio.values.tolist()

    def save_model(self, filename):
        filedir = Path(filename).parent.resolve()
        filedir.mkdir(mode=0o755, parents=True, exist_ok=True)
        filepath = Path(filename).resolve()
        logger.debug(f"saving the model to {filepath}")
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'thresholds': self.thresholds,
        }, filename)


class NoniidDistributedTrainEnvironment(NoniidSingleTrainEnvironment):

    def __init__(self, device, local_rank, amp_enable=False):
        rank = dist.get_rank()
        dataset_id = rank % len(DATASETS)

        super().__init__(device, train_data=DATASETS[dataset_id], amp_enable=amp_enable)
        self.distributed = True
        self.local_rank = local_rank
        self.world_size = dist.get_world_size()
        self.rank = rank
        logger.info(f"initialized on {device} as rank {self.rank} of {self.world_size}")

        if dataset_id == 0:
            train_sets = [self.stanford_datasets[0]]
            test_sets = [self.stanford_datasets[1]]
            if self.mode == "per_study":
                batch_size = 7
            else:
                batch_size = 12
        elif dataset_id == 1:
            train_sets = [self.mimic_datasets[0]]
            test_sets = [self.mimic_datasets[1]]
            if self.mode == "per_study":
                batch_size = 8
            else:
                batch_size = 21
        else: # dataset_id == 2
            train_sets = [self.nih_datasets[0]]
            test_sets = [self.nih_datasets[1]]
            if self.mode == "per_study":
                batch_size = 4
            else:
                batch_size = 6

        self.set_data_loader(train_sets, test_sets, batch_size=batch_size)

        #self.model = DistributedDataParallel(self.model, device_ids=[self.device],
        #                                     output_device=self.device, find_unused_parameters=True)
        self.model.to_distributed(self.device)

        self.positive_weights = torch.FloatTensor(self.get_positive_weights()).to(device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.positive_weights, reduction='none')
        #self.loss = nn.BCEWithLogitsLoss(reduction='none')


class NoniidTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, num_epoch, start_epoch=1):
        if start_epoch > 1:
            model_path = runtime_path.joinpath(f"model_epoch_{(start_epoch - 1):03d}.{self.env.rank}.pth.tar")
            self.env.load_model(model_path)
            self.load()

        for epoch in range(start_epoch, num_epoch + 1):
            self.train_epoch(epoch)
            for i, test_loader in enumerate(self.env.test_loaders):
                prefix = f"test{i}_"
                ys, ys_hat = self.test(epoch, test_loader, prefix=prefix)
                self.calculate_metrics(epoch, ys, ys_hat, prefix)
            self.save()
            if self.env.scheduler is not None:
                self.env.scheduler.step()
                logger.info(f"lr = {self.env.scheduler.get_lr()}")

    def cross_test_only(self, num_epoch, start_epoch=1):
        self.load()
        for epoch in range(start_epoch, num_epoch + 1):
            model_path = runtime_path.joinpath(f"model_epoch_{epoch:03d}.pth.tar")
            try:
                self.env.load_model(model_path)
                for i, test_loader in enumerate(self.env.test_loaders):
                    self.test(epoch, test_loader, prefix=f"test{i}_")
                self.save()
            except:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CXR Training")
    # for training
    parser.add_argument('--cuda', default=None, type=str, help="use GPUs with its device ids, separated by commas")
    parser.add_argument('--amp', default=False, action='store_true', help="use automatic mixed precision for faster training")
    parser.add_argument('--epoch', default=500, type=int, help="max number of epochs")
    parser.add_argument('--start-epoch', default=1, type=int, help="start epoch, especially need to continue from a stored model")
    parser.add_argument('--runtime-dir', default='./runtime', type=str, help="runtime directory to store log, pretrained models, and tensorboard metadata")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="true if logging to tensorboard")
    parser.add_argument('--slack', default=False, action='store_true', help="true if logging to slack")
    parser.add_argument('--main-dataset', default='stanford', type=str, help="main dataset for training (single mode only)")
    parser.add_argument('--local_rank', default=None, type=int, help="this is for the use of torch.distributed.launch utility")
    parser.add_argument('--ignore-repo-dirty', default=False, action='store_true', help="not checking the repo clean")
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

