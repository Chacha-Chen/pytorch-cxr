from pathlib import Path

from tqdm import tqdm
import numpy as np
import torchnet as tnt

from utils import logger, print_versions, get_devices
from train import TrainEnvironment, Trainer
from eth import Web3Connector


class EthTrainEnvironment(TrainEnvironment):

    def __init__(self, devices, local_rank):
        super().__init__(devices[local_rank])

        self.local_rank = local_rank
        self.distributed = True

        if self.device != torch.device("cpu"):
            torch.cuda.set_device(self.device)

        conn = Web3Connector(eth_uri)
        logger.info(f"initialized on {self.device} as rank {conn.rank} of {conn.world_size}")

        #TODO: do we need to do this?
        self.model = DistributedDataParallel(self.model, device_ids=[self.device],
                                             output_device=self.device, find_unused_parameters=True)
        train_set = self.train_loader.dataset
        batch_size = self.train_loader.batch_size
        num_workers = self.train_loader.num_workers
        self.train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                                       sampler=DistributedSampler(train_set), shuffle=False, pin_memory=True)

    def save_data_indices(self, runtime_path):
        # save only if local_rank == 0
        if self.local_rank == 0:
            super().save_data_indices(runtime_path)

    def save_model(self, filename):
        # save only if local_rank == 0
        if self.local_rank == 0:
            super().save_model(filename)


class EthTrainer(Trainer):

    def train_epoch(self, epoch):
        train_loader = self.env.train_loader
        train_set = train_loader.dataset

        self.env.model.train()
        progress = 0

        ave_len = len(train_loader) // 100 + 1
        ave_loss = tnt.meter.MovingAverageValueMeter(ave_len)

        ckpt_step = 0.1
        ckpts = iter(len(train_set) * np.arange(ckpt_step, 1 + ckpt_step, ckpt_step))
        ckpt = next(ckpts)

        if self.env.distributed:
            tqdm_desc = f"training{self.env.local_rank}"
            tqdm_pos = self.env.local_rank
        else:
            tqdm_desc = "training"
            tqdm_pos = 0

        t = tqdm(enumerate(train_loader), total=len(train_loader), desc=tqdm_desc,
                 dynamic_ncols=True, position=tqdm_pos)

        for batch_idx, (data, target) in t:
            data, target = data.to(self.env.device), target.to(self.env.device)

            self.env.optimizer.zero_grad()
            output = self.env.model(data)
            loss = self.env.loss(output, target)
            loss.backward()
            self.env.optimizer.step()

            #self.env.scheduler.step(loss.item())

            t.set_description(f"{tqdm_desc} (loss: {loss.item():.4f})")
            t.refresh()

            ave_loss.add(loss.item())
            progress += len(data)

            if progress > ckpt and self.tensorboard:
                #logger.info(f"train epoch {epoch:03d}:  "
                #        f"progress/total {progress:06d}/{len(train_loader.dataset):06d} "
                #            f"({100. * batch_idx / len(train_loader):6.2f}%)  "
                #            f"loss {loss.item():.6f}")

                x = (epoch - 1) + progress / len(train_set)
                global_step = int(x / ckpt_step)
                self.writer.add_scalar("loss", loss.item(), global_step=global_step)
                ckpt = next(ckpts)

            del loss

        logger.info(f"train epoch {epoch:03d}:  "
                    f"loss {ave_loss.value()[0]:.6f}")

        self.env.save_model(self.runtime_path.joinpath(f"model_epoch_{epoch:03d}.pth.tar"))


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="CXR Training on Ethereum Network")
    # for training on ethereum network
    parser.add_argument('--uri', default="http://localhost:9545", type=str, help="Eth RPC URI to connect")
    parser.add_argument('--cuda', default=None, type=str, help="use GPUs with its device ids, separated by commas")
    parser.add_argument('--epoch', default=100, type=int, help="max number of epochs")
    parser.add_argument('--start-epoch', default=1, type=int, help="start epoch, especially need to continue from a stored model")
    parser.add_argument('--runtime-dir', default='./runtime', type=str, help="runtime directory to store log, pretrained models, and tensorboard metadata")
    parser.add_argument('--tensorboard', default=False, action='store_true', help="true if logging to tensorboard")
    parser.add_argument('--slack', default=False, action='store_true', help="true if logging to slack")
    parser.add_argument('--local_rank', default=None, type=int, help="this is for the use of torch.distributed.launch utility")
    args = parser.parse_args()

    runtime_path = Path(args.runtime_dir).resolve()

    # set logger
    if args.local_rank is None:
        log_file = "train.log"
        logger.set_log_to_stream()
    else:
        log_file = f"train.{args.local_rank}.log"
        logger.set_log_to_stream(level=logging.INFO)

    logger.set_log_to_file(runtime_path.joinpath(log_file))
    if args.slack:
        set_log_to_slack(Path(__file__).parent.joinpath(".slack"), runtime_path.name)

    # print versions after logger.set_log_to_file() to log them into file
    print_versions()
    logger.info(f"runtime_path: {runtime_path}")

    # check eth uri
    eth_uri = os.environ.get('WEB3_PROVIDER_URI', args.uri)

    # start training
    devices = get_devices(args.cuda)
    env = EthTrainEnvironment(eth_uri, devices, args.local_rank)
    t = EthTrainer(env, runtime_path=runtime_path, tensorboard=args.tensorboard)
    t.train(args.epoch, start_epoch=args.start_epoch)

