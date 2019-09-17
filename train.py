from pathlib import Path
import pickle
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
import sklearn.metrics as sklm

from apex import amp

from utils import logger, print_versions, get_devices, get_ip, get_commit
from adamw import AdamW
from predict import PredictEnvironment, Predictor
from dataset import STANFORD_CXR_BASE, MIMIC_CXR_BASE, NIH_CXR_BASE, CxrDataset, CxrConcatDataset, CxrSubset, cxr_random_split


def check_distributed(args):
    devices = get_devices(args.cuda)
    if args.local_rank is None:
        return False, devices[0]
    else:
        device = devices[args.local_rank % len(devices)]
        torch.cuda.set_device(device)
        logger.info(f"waiting other ranks ...")
        dist.init_process_group(backend="nccl", init_method="env://")
        logger.info(f"set world_size to {dist.get_world_size()}")
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

        mode = "per_study"
        stanford_train_set = CxrDataset(STANFORD_CXR_BASE, "train.csv", mode=mode)
        stanford_test_set = CxrDataset(STANFORD_CXR_BASE, "valid.csv", mode=mode)

        #mimic_train_set = CxrDataset(MIMIC_CXR_BASE, "train.csv", mode=mode)
        #mimic_test_set = CxrDataset(MIMIC_CXR_BASE, "valid.csv", mode=mode)

        #concat_set = CxrConcatDataset([stanford_train_set, stanford_test_set, mimic_train_set, mimic_test_set])
        #concat_set = CxrConcatDataset([stanford_train_set, stanford_test_set])

        #datasets = cxr_random_split(concat_set, [360000, 15000])
        #datasets = cxr_random_split(concat_set, [400, 200])
        #subset = Subset(concat_set, range(0, 36))
        #datasets = random_split(subset, [len(subset) - 12, 12])

        datasets = [stanford_train_set, stanford_test_set]

        pin_memory = True if self.device.type == 'cuda' else False
        self.train_loader = DataLoader(datasets[0], batch_size=32, num_workers=8, shuffle=True, pin_memory=pin_memory)
        self.test_loader = DataLoader(datasets[1], batch_size=32, num_workers=8, shuffle=False, pin_memory=pin_memory)

        self.labels = [x.lower() for x in datasets[0].labels]
        self.out_dim = len(self.labels)
        self.positive_weights = torch.FloatTensor(self.get_positive_weights()).to(device)

        #img, tar = datasets[0]
        #plt.imshow(img.squeeze(), cmap='gray')

        super().__init__(out_dim=self.out_dim, device=self.device, mode=mode)

        self.optimizer = AdamW(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
        self.scheduler = None
        #self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.positive_weights, reduction='none')
        #self.loss = nn.BCEWithLogitsLoss(reduction='none')

        if self.amp:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def get_positive_weights(self):
        df = self.train_loader.dataset.get_label_counts()
        ratio = df.loc[0] / df.loc[1]
        return ratio.values.tolist()

    def load_model(self, filename):
        ckpt = super().load_model(filename)
        if 'optimizer_state' in ckpt:
            optimizer_state = ckpt['optimizer_state']
            self.optimizer.load_state_dict(optimizer_state)

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
        #self.test_loader = DataLoader(self.test_loader.dataset,
        #                              batch_size=self.test_loader.batch_size,
        #                              num_workers=self.test_loader.num_workers,
        #                              sampler=DistributedSampler(self.test_loader.dataset),
        #                              shuffle=False, pin_memory=pin_memory)

        self.positive_weights = torch.FloatTensor(self.get_positive_weights()).to(device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.positive_weights, reduction='none')
        #self.loss = nn.BCEWithLogitsLoss(reduction='none')

class Trainer:

    def __init__(self, env, runtime_path="train", tensorboard=False):
        self.env = env
        self.runtime_path = runtime_path
        self.tensorboard = tensorboard
        if tensorboard:
            tblog_path = runtime_path.joinpath(f"tensorboard.{self.env.rank}").resolve()
            tblog_path.mkdir(mode=0o755, parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tblog_path))
            #self.writer.add_custom_scalars({'Stuff': {
            #    'Losses': ['MultiLine', ['loss/(one|two)']],
            #    'Metrics': ['MultiLine', ['metric/(three|four)']],
            #}})

        self.metrics = {}

        train_set_percent = len(self.env.train_loader.sampler) / len(self.env.train_loader.dataset) * 100.
        test_set_percent = len(self.env.test_loader.sampler) / len(self.env.test_loader.dataset) * 100.
        logger.info(f"using {len(self.env.train_loader.sampler)}/{len(self.env.train_loader.dataset)} ({train_set_percent:.1f}%) entries for training")
        logger.info(f"using {len(self.env.test_loader.sampler)}/{len(self.env.test_loader.dataset)} ({test_set_percent:.1f}%) entries for testing")

    def add_metric(self, keystr, point):
        keys = keystr.split('/')

        def add_key(node, keys, point):
            if not keys:
                node.append(point)
                return
            if keys[0] in node:
                add_key(node[keys[0]], keys[1:], point)
            else:
                node[keys[0]] = [] if len(keys) == 1 else {}
                add_key(node[keys[0]], keys[1:], point)

        add_key(self.metrics, keys, point)

    def train(self, num_epoch, start_epoch=1):
        if start_epoch > 1:
            model_path = runtime_path.joinpath(f"model_epoch_{(start_epoch - 1):03d}.{self.env.rank}.pth.tar")
            self.env.load_model(model_path)
            if self.env.distributed:
                self.env.train_loader.sampler.set_epoch(start_epoch - 1)
            self.load()

        for epoch in range(start_epoch, num_epoch + 1):
            self.train_epoch(epoch)
            ys, ys_hat = self.test(epoch, self.env.test_loader)
            self.calculate_metrics(epoch, ys, ys_hat)
            self.save()

    def train_epoch(self, epoch, ckpt=False):
        labels = self.env.labels
        train_loader = self.env.train_loader
        train_set = train_loader.dataset

        CxrDataset.train()
        self.env.model.train()

        ave_len = len(train_loader) // 100 + 1
        ave_losses = [tnt.meter.MovingAverageValueMeter(ave_len) for _ in labels]
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

        for batch_idx, (data, target, input_length) in t:
            data, target = data.to(self.env.device), target.to(self.env.device)
            self.env.optimizer.zero_grad()
            output = self.env.model(data, input_length)
            losses = self.env.loss(output, target).mean(dim=0)
            loss = losses.mean()
            for a, l in zip(ave_losses, losses):
                a.add(l.item())
            ave_loss.add(loss.item())
            if self.env.amp:
                with amp.scale_loss(loss, self.env.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            #nn.utils.clip_grad_norm_(self.env.model.parameters(), 1e-2)
            self.env.optimizer.step()

            if self.env.scheduler is not None:
                self.env.scheduler.step(loss.item())

            t.set_description(f"{tqdm_desc} (loss: {ave_loss.value()[0].item():.4f})")
            t.refresh()

            if ckpt and progress > ckpt and self.tensorboard:
                progress += len(data)
                x = (epoch - 1) + progress / len(train_set)
                global_step = int(x / ckpt_step)
                for i, l in enumerate(labels):
                    self.writer.add_scalar(f"{l}/loss", ave_losses[i].value()[0].item(), global_step=global_step)
                self.writer.add_scalar("total/loss", ave_loss.value()[0].item(), global_step=global_step)
                ckpt = next(ckpts)

            del loss

        logger.info(f"train epoch {epoch:03d}:  "
                    f"ave loss {ave_loss.value()[0].item():.6f}")

        p = PrettyTable()
        p.field_names = ["labels", f"loss (ave. {ave_loss.value()[0].item():.6f})"]
        for i, l in enumerate(labels):
            p.add_row([l, f"{ave_losses[i].value()[0].item():.6f}"])
        tbl_str = p.get_string()  #title=f"metrics per label")
        logger.info(f"\n{tbl_str}")

        if not ckpt and self.tensorboard:
            self.writer.add_scalar("total/loss", ave_loss.value()[0].item(), global_step=epoch)
            for i, l in enumerate(labels):
                self.writer.add_scalar(f"{l}/loss", ave_losses[i].value()[0].item(), global_step=epoch)

        self.add_metric('total/loss', (epoch, ave_loss.value()[0].item()))
        for i, l in enumerate(labels):
            self.add_metric(f"{l}/loss", (epoch, ave_losses[i].value()[0].item()))

        self.env.save_model(self.runtime_path.joinpath(f"model_epoch_{epoch:03d}.{self.env.rank}.pth.tar"))

    def test(self, epoch, test_loader, prefix=""):
        out_dim = self.env.out_dim

        CxrDataset.eval()
        self.env.model.eval()
        with torch.no_grad():
            tqdm_desc = f"{prefix}testing [{self.env.rank}]"
            tqdm_pos = self.env.local_rank

            t = tqdm(enumerate(test_loader), total=len(test_loader), desc=tqdm_desc,
                     dynamic_ncols=True, position=tqdm_pos)

            ys_hat = np.empty(shape=[0, out_dim])
            ys = np.empty(shape=[0, out_dim])

            for batch_idx, (data, target, input_length) in t:
                data, target = data.to(self.env.device), target.to(self.env.device)
                output = self.env.model(data, input_length)

                ys = np.append(ys, target.cpu().numpy(), axis=0)
                ys_hat = np.append(ys_hat, output.cpu().numpy(), axis=0)

        return ys, ys_hat

    def calculate_metrics(self, epoch, ys, ys_hat, prefix="", roc=False, prc=False):
        out_dim = self.env.out_dim
        labels = self.env.labels

        # roc and threshold
        selected_tprs = [0.] * out_dim
        for i, l in enumerate(labels):
            fprs, tprs, thrs = sklm.roc_curve(ys[:, i], ys_hat[:, i])
            rects = [(1. - w) * h for w, h in zip(fprs, tprs)]
            idx = np.argmax(rects)
            selected_tprs[i] = tprs[idx]
            self.env.thresholds[i] = thrs[idx]
            if roc and self.tensorboard:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(fprs, tprs, 'b-')
                ax1.plot([0., 1.], [0., 1.], 'k--')
                ax1.plot([fprs[idx], fprs[idx]], [0., 1.], 'c--')
                ax1.plot([0., 1.], [tprs[idx], tprs[idx]], 'c--')
                ax1.set_xlim([0., 1.])
                ax1.set_ylim([0., 1.])
                plt.xlabel('FPR (1-specificity)')
                plt.ylabel('TPR (sensitivity, recall)')
                plt.title(f'ROC curve for {l} at epoch {epoch}')
                ax2 = ax1.twinx()
                ax2.plot(fprs, thrs, 'r-')
                ax2.plot([0., 1.], [thrs[idx], thrs[idx]], 'm--')
                ax2.set_ylim([thrs[-1], thrs[0]])
                ax2.set_ylabel('threshold')
                self.writer.add_figure(f"{l}/{prefix}roc_curve", fig, global_step=epoch)
            self.add_metric(f'{l}/{prefix}roc_curve', (epoch, (fprs, tprs, thrs)))

        # accuracy
        accuracies = [0.] * out_dim
        summary = {}
        for i, l in enumerate(labels):
            t, p = ys[:, i].astype(np.int).flatten(), (ys_hat[:, i] > self.env.thresholds[i]).astype(np.int).flatten()
            accuracies[i] = sklm.accuracy_score(t, p, normalize=True)
            summary[l] = sklm.confusion_matrix(t, p).ravel().tolist()
        total_accuracy = (np.sum(accuracies) * ys.shape[0]) / ys.size

        df = pd.DataFrame(summary, index=['tn', 'fp', 'fn', 'tp'])
        logger.info(f"decision result:\n{df}")
        logger.info(f"val epoch {epoch:03d}:  "
                    f"{prefix}accuracy {total_accuracy:.6f}")

        if self.tensorboard:
            self.writer.add_scalar(f"total/{prefix}accuracy", total_accuracy, global_step=epoch)
            for i, l in enumerate(labels):
                self.writer.add_scalar(f"{l}/{prefix}accuracy", accuracies[i], global_step=epoch)

        self.add_metric(f'total/{prefix}accuracy', (epoch, total_accuracy))
        for i, l in enumerate(labels):
            self.add_metric(f'{l}/{prefix}accuracy', (epoch, accuracies[i]))

        # auc score
        auc_scores = sklm.roc_auc_score(ys, ys_hat, average=None)
        average_auc_score = np.mean(auc_scores)

        p = PrettyTable()
        p.field_names = ["labels", f"accuracy (tot. {total_accuracy:.6f})", f"auc_score (ave. {average_auc_score:.6f})"]
        for i, l in enumerate(labels):
            p.add_row([l, f"{accuracies[i]:.6f}", f"{auc_scores[i]:.6f}"])
        tbl_str = p.get_string()  #title=f"{prefix}metrics per label")
        logger.info(f"\n{tbl_str}")

        if self.tensorboard:
            self.writer.add_scalar(f"total/{prefix}average_auc_score", average_auc_score, global_step=epoch)
            for i, l in enumerate(labels):
                self.writer.add_scalar(f"{l}/{prefix}auc_score", auc_scores[i], global_step=epoch)

        self.add_metric(f'total/{prefix}auc_score', (epoch, average_auc_score))
        for i, l in enumerate(labels):
            self.add_metric(f'{l}/{prefix}auc_score', (epoch, auc_scores[i]))

        if not prc:
            return

        # precision and recall
        for i, l in enumerate(labels):
            pcs, rcs, thrs = sklm.precision_recall_curve(ys[:, i], ys_hat[:, i])
            rects = [w * h for w, h in zip(rcs, pcs)]
            idx = np.argmax(rects)
            if self.tensorboard:
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(rcs, pcs, 'b-')
                ax1.plot([rcs[idx], rcs[idx]], [0., 1.], 'c--')
                ax1.plot([0., 1.], [pcs[idx], pcs[idx]], 'c--')
                ax1.plot([selected_tprs[i], selected_tprs[i]], [0., 1.], 'g--')
                ax1.set_xlim([0., 1.])
                ax1.set_ylim([0., 1.])
                plt.xlabel('recall (TPR, sensitivity)')
                plt.ylabel('precision')
                plt.title(f'precision-recall curve for {l} at epoch {epoch}')
                ax2 = ax1.twinx()
                ax2.plot(rcs[:-1], thrs, 'r-')
                ax2.plot([0., 1.], [thrs[idx], thrs[idx]], 'm--')
                ax2.plot([0., 1.], [self.env.thresholds[i], self.env.thresholds[i]], 'g--')
                ax2.set_ylim([thrs[0], thrs[-1]])
                ax2.set_ylabel('threshold')
                self.writer.add_figure(f"{l}/{prefix}precision_recall_curve", fig, global_step=epoch)
            self.add_metric(f'{l}/{prefix}precision_recall_curve', (epoch, (rcs, pcs, thrs)))

            t, p = ys[:, i].astype(np.int), (ys_hat[:, i] > self.env.thresholds[i]).astype(np.int)
            precision, recall, f1_score, support = sklm.precision_recall_fscore_support(t, p, beta=1.0, labels=[1, 0])
            if self.tensorboard:
                self.writer.add_scalar(f"{l}/{prefix}precision", precision[0], global_step=epoch)
                self.writer.add_scalar(f"{l}/{prefix}recall", recall[0], global_step=epoch)
                self.writer.add_scalar(f"{l}/{prefix}f1_score", f1_score[0], global_step=epoch)
            self.add_metric(f'{l}/{prefix}precision', (epoch, precision[0]))
            self.add_metric(f'{l}/{prefix}recall', (epoch, recall[0]))
            self.add_metric(f'{l}/{prefix}f1_score', (epoch, f1_score[0]))

    def load(self):
        filepath = self.runtime_path.joinpath(f"train.{self.env.rank}.pkl")
        with open(filepath, 'rb') as f:
            self.metrics = pickle.load(f)

    def save(self):
        filepath = self.runtime_path.joinpath(f"train.{self.env.rank}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self.metrics, f)


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
    #logger.set_log_to_stream()
    logger.set_log_to_file(runtime_path.joinpath(log_file))
    if args.slack:
        logger.set_log_to_slack(Path(__file__).parent.joinpath(".slack"), runtime_path.name)

    # print versions after logger.set_log_to_file() to log them into file
    print_versions()
    run_mode = "distributed" if distributed else "single"
    logger.info(f"runtime node: {get_ip()} ({run_mode}, rank: {rank}, local_rank: {local_rank})")
    logger.info(f"runtime commit: {get_commit(args.ignore_repo_dirty)}")
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
    parser.add_argument('--ignore-repo-dirty', default=False, action='store_true', help="not checking the repo clean")
    args = parser.parse_args()

    distributed, runtime_path, device = initialize(args)

    # start training
    env = DistributedTrainEnvironment(device, args.local_rank, amp_enable=args.amp) if distributed else \
          TrainEnvironment(device, amp_enable=args.amp)

    t = Trainer(env, runtime_path=runtime_path, tensorboard=args.tensorboard)
    t.train(args.epoch, start_epoch=args.start_epoch)

