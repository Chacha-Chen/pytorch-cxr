from pathlib import Path
import pickle

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
from dataset import STANFORD_CXR_BASE, MIMIC_CXR_BASE, CxrDataset, CxrConcatDataset, CxrSubset, cxr_random_split
from utils import logger, print_versions, get_devices, get_ip, get_commit
from adamw import AdamW


def check_distributed(args):
    devices = get_devices(args.cuda)
    if args.local_rank is None:
        return False, devices[0]
    else:
        device = devices[args.local_rank]
        torch.cuda.set_device(device)
        logger.info(f"waiting other ranks ...")
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, device


class TrainEnvironment(PredictEnvironment):
    """
    This environment object inherits PredictEnvironment from predict.py,
    with adding the datasets and their corresponding data loaders,
    optimizer, lr scheduler, and the loss function using in training.
    """

    def __init__(self, device, amp_enable=False):
        self.device = device
        self.distributed = False
        self.amp = amp_enable

        self.local_rank = 0
        self.rank = 0

        stanford_train_set = CxrDataset(STANFORD_CXR_BASE, "train.csv", mode="per_study")
        stanford_test_set = CxrDataset(STANFORD_CXR_BASE, "valid.csv", mode="per_study")

        mimic_train_set = CxrDataset(MIMIC_CXR_BASE, "train.csv", mode="per_study")
        mimic_test_set = CxrDataset(MIMIC_CXR_BASE, "valid.csv", mode="per_study")

        concat_set = CxrConcatDataset([stanford_train_set, stanford_test_set, mimic_train_set, mimic_test_set])

        datasets = cxr_random_split(concat_set, [360000, 15000])
        #datasets = cxr_random_split(concat_set, [40000, 20000])
        #subset = Subset(concat_set, range(0, 36))
        #datasets = random_split(subset, [len(subset) - 12, 12])

        pin_memory = True if self.device.type == 'cuda' else False
        self.train_loader = DataLoader(datasets[0], batch_size=32, num_workers=16, shuffle=True, pin_memory=pin_memory)
        self.test_loader = DataLoader(datasets[1], batch_size=32, num_workers=16, shuffle=False, pin_memory=pin_memory)

        self.labels = datasets[0].labels
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


class DistributedTrainEnvironment(TrainEnvironment):

    def __init__(self, device, local_rank, amp_enable=False):
        super().__init__(device, amp_enable=amp_enable)
        self.distributed = True
        self.local_rank = local_rank
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        logger.info(f"initialized on {device} as rank {self.rank} of {self.world_size}")

        self.model = DistributedDataParallel(self.model, device_ids=[self.device],
                                             output_device=self.device, find_unused_parameters=True)

        pin_memory = True if self.device.type == 'cuda' else False
        self.train_loader = DataLoader(self.train_loader.dataset,
                                       batch_size=self.train_loader.batch_size,
                                       num_workers=self.train_loader.num_workers,
                                       sampler=DistributedSampler(self.train_loader.dataset),
                                       shuffle=False, pin_memory=pin_memory)
        self.test_loader = DataLoader(self.test_loader.dataset,
                                      batch_size=self.test_loader.batch_size,
                                      num_workers=self.test_loader.num_workers,
                                      sampler=DistributedSampler(self.test_loader.dataset),
                                      shuffle=False, pin_memory=pin_memory)


class Trainer:

    def __init__(self, env, runtime_path="train", tensorboard=False):
        self.env = env
        self.runtime_path = runtime_path
        self.tensorboard = tensorboard
        if tensorboard:
            tblog_path = runtime_path.joinpath("tensorboard").resolve()
            tblog_path.mkdir(mode=0o755, parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tblog_path))
        self.progress = {
            'loss': [],
            'accuracy': [],
            'auc_score': [],
        }

        train_set_percent = len(self.env.train_loader.sampler) / len(self.env.train_loader.dataset) * 100.
        test_set_percent = len(self.env.test_loader.sampler) / len(self.env.test_loader.dataset) * 100.
        logger.info(f"using {len(self.env.train_loader.sampler)}/{len(self.env.train_loader.dataset)} ({train_set_percent:.1f}%) entries for training")
        logger.info(f"using {len(self.env.test_loader.sampler)}/{len(self.env.test_loader.dataset)} ({test_set_percent:.1f}%) entries for testing")

    def train(self, num_epoch, start_epoch=1):
        if start_epoch > 1:
            model_path = runtime_path.joinpath(f"model_epoch_{(start_epoch - 1):03d}.{self.env.rank}.pth.tar")
            self.env.load_model(model_path)
            if self.env.distributed:
                self.env.train_loader.sampler.set_epoch(start_epoch - 1)
            self.load()

        for epoch in range(start_epoch, num_epoch + 1):
            self.train_epoch(epoch)
            self.test(epoch, self.env.test_loader)
            self.save()

    def train_epoch(self, epoch, ckpt=False):
        train_loader = self.env.train_loader
        train_set = train_loader.dataset

        CxrDataset.train()
        self.env.model.train()

        ave_len = len(train_loader) // 100 + 1
        ave_loss = tnt.meter.MovingAverageValueMeter(ave_len)

        if ckpt:
            progress = 0
            ckpt_step = 0.1
            ckpts = iter(len(train_set) * np.arange(ckpt_step, 1 + ckpt_step, ckpt_step))
            ckpt = next(ckpts)

        tqdm_desc = f"training [{self.env.rank}]"
        tqdm_pos = self.env.local_rank

        t = tqdm(enumerate(train_loader), total=len(train_loader), desc=tqdm_desc,
                 dynamic_ncols=True, position=tqdm_pos)

        for batch_idx, (data, target) in t:
            data, target = data.to(self.env.device), target.to(self.env.device)

            self.env.optimizer.zero_grad()
            output = self.env.model(data)
            loss = self.env.loss(output, target)
            if self.env.amp:
                with amp.scale_loss(loss, self.env.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.env.optimizer.step()

            #self.env.scheduler.step(loss.item())

            t.set_description(f"{tqdm_desc} (loss: {loss.item():.4f})")
            t.refresh()
            ave_loss.add(loss.item())

            if ckpt and progress > ckpt and self.tensorboard:
                progress += len(data)
                #logger.info(f"train epoch {epoch:03d}:  "
                #        f"progress/total {progress:06d}/{len(train_loader.dataset):06d} "
                #            f"({100. * batch_idx / len(train_loader):6.2f}%)  "
                #            f"loss {loss.item():.6f}")

                x = (epoch - 1) + progress / len(train_set)
                global_step = int(x / ckpt_step)
                self.writer.add_scalar("loss", loss.item(), global_step=global_step)
                ckpt = next(ckpts)

            del loss

        if not ckpt and self.tensorboard:
            self.writer.add_scalar("loss", ave_loss.value()[0].item(), global_step=epoch)

        self.progress['loss'].append((epoch, ave_loss.value()[0].item()))
        logger.info(f"train epoch {epoch:03d}:  "
                    f"ave loss {ave_loss.value()[0]:.6f}")

        self.env.save_model(self.runtime_path.joinpath(f"model_epoch_{epoch:03d}.{self.env.rank}.pth.tar"))

    def test(self, epoch, test_loader, prefix=""):
        test_set = test_loader.dataset
        out_dim = self.env.out_dim
        labels = self.env.labels

        aucs = [tnt.meter.AUCMeter() for i in range(out_dim)]

        CxrDataset.eval()
        self.env.model.eval()
        with torch.no_grad():
            test_loss = 0
            correct = 0

            tqdm_desc = f"{prefix}testing [{self.env.rank}]"
            tqdm_pos = self.env.local_rank

            t = tqdm(enumerate(test_loader), total=len(test_loader), desc=tqdm_desc,
                     dynamic_ncols=True, position=tqdm_pos)

            ones = torch.ones((out_dim)).int().to(self.env.device)
            zeros = torch.zeros((out_dim)).int().to(self.env.device)

            for batch_idx, (data, target) in t:
                data, target = data.to(self.env.device), target.to(self.env.device)
                output = self.env.model(data)
                for i in range(out_dim):
                    aucs[i].add(output[:, i], target[:, i])
                pred = torch.where(output > 0., ones, zeros)
                correct += pred.eq(target.int()).sum().item()

            #correct /= out_dim
            total = len(test_loader.sampler) * out_dim
            percent = 100. * correct / total
            if self.tensorboard:
                self.writer.add_scalar(f"{prefix}accuracy", percent, global_step=epoch)

            logger.info(f"val epoch {epoch:03d}:  "
                        f"{prefix}accuracy {correct}/{total} "
                        f"({percent:.2f}%)")

            p = PrettyTable()
            p.field_names = ["findings", "score"]
            for i in range(out_dim):
                p.add_row([labels[i], f"{aucs[i].value()[0]:.6f}"])
            ave_auc = np.mean([k.value()[0] for k in aucs])
            tbl_str = p.get_string(title=f"{prefix}auc scores (average {ave_auc:.6f})")
            if self.tensorboard:
                self.writer.add_scalar(f"{prefix}auc_score", ave_auc, global_step=epoch)
            logger.info(f"\n{tbl_str}")

        self.progress[f'{prefix}accuracy'].append((epoch, correct / total))
        self.progress[f'{prefix}auc_score'].append((epoch, ave_auc))

    def load(self):
        filepath = self.runtime_path.joinpath(f"train.{self.env.rank}.pkl")
        with open(filepath, 'rb') as f:
            self.progress = pickle.load(f)

    def save(self):
        filepath = self.runtime_path.joinpath(f"train.{self.env.rank}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.progress, f)


# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.

def convert_image_np(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn():
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()
        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))
        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))
        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')
        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def initialize(args):
    if args.amp:
        assert args.cuda is not None

    runtime_path = Path('./runtime', args.runtime_dir).resolve()
    #runtime_path = Path("train_20190527_per_study_256").resolve()

    # check if distributed or not
    distributed, device = check_distributed(args)

    # set logger
    local_rank = args.local_rank if distributed else 0
    rank = dist.get_rank() if distributed else 0
    logger.set_rank(rank)

    log_file = f"train.{rank}.log"
    logger.set_log_to_stream()
    logger.set_log_to_file(runtime_path.joinpath(log_file))
    if args.slack:
        logger.set_log_to_slack(Path(__file__).parent.joinpath(".slack"), runtime_path.name)

    # print versions after logger.set_log_to_file() to log them into file
    print_versions()
    run_mode = "distributed" if distributed else "single"
    logger.info(f"runtime node: {get_ip()} ({run_mode}, rank: {rank}, local_rank: {local_rank})")
    logger.info(f"runtime commit: {get_commit()}")
    logger.info(f"runtime path: {runtime_path}")

    # for fixed random indices
    torch.manual_seed(2019)
    np.random.seed(2019)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return distributed, runtime_path, device


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CXR Training")
    # for training
    parser.add_argument('--cuda', default=None, type=str, help="use GPUs with its device ids, separated by commas")
    parser.add_argument('--amp', default=False, action='store_true', help="use automatic mixed precision for faster training")
    parser.add_argument('--epoch', default=100, type=int, help="max number of epochs")
    parser.add_argument('--start-epoch', default=1, type=int, help="start epoch, especially need to continue from a stored model")
    parser.add_argument('--runtime-dir', default='current', type=str, help="runtime directory to store log, pretrained models, and tensorboard metadata")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="true if logging to tensorboard")
    parser.add_argument('--slack', default=False, action='store_true', help="true if logging to slack")
    parser.add_argument('--local_rank', default=None, type=int, help="this is for the use of torch.distributed.launch utility")
    args = parser.parse_args()

    distributed, runtime_path, device = initialize(args)

    # start training
    env = DistributedTrainEnvironment(device, args.local_rank, amp_enable=args.amp) if distributed else \
          TrainEnvironment(device, amp_enable=args.amp)

    t = Trainer(env, runtime_path=runtime_path, tensorboard=args.tensorboard)
    t.train(args.epoch, start_epoch=args.start_epoch)

    # Visualize the STN transformation on some input batch
    #visualize_stn()
    #plt.show()
