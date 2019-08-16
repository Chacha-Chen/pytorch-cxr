import os
import sys
import logging
from pathlib import Path
import yaml

import torch
import torchvision

# in order to avoid complaining warning from tensorflow logger
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False


class SlackClientHandler(logging.Handler):

    def __init__(self, credential_file, ch_name):
        super().__init__()
        with open(credential_file, 'r') as f:
            tmp = yaml.safe_load(f)
        self.slack_token = tmp['token']
        self.slack_recipients = tmp['recipients']
        #self.slack_token = os.getenv("SLACK_API_TOKEN")
        #self.slack_user = os.getenv("SLACK_API_USER")
        if self.slack_token is None or self.slack_recipients is None:
            raise KeyError

        from slack import WebClient
        self.client = WebClient(self.slack_token)

        # getting user id
        ans = self.client.users_list()
        users = [u['id'] for u in ans['members'] if u['name'] in self.slack_recipients]
        # open DM channel to the users
        ans = self.client.conversations_open(users=','.join(users))
        self.channel = ans['channel']['id']
        ans = self.client.chat_postMessage(channel=self.channel, text=f"*{ch_name}*")
        self.thread = ans['ts']

    def emit(self, record):
        try:
            msg = self.format(record)
            self.client.chat_postMessage(channel=self.channel, thread_ts=self.thread, text=f"```{msg}```")
        except:
            self.handleError(record)


class MyFilter(logging.Filter):

    def __init__(self, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank

    def filter(self, record):
        record.rank = self.rank
        return True


class MyLogger(logging.Logger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addFilter(MyFilter('-'))
        self.formatter = logging.Formatter('%(asctime)s %(rank)s [%(levelname)-5s] %(message)s')
        self.set_log_to_stream(level=logging.DEBUG)

    def set_rank(self, rank):
        self.removeFilter(self.filter)
        self.addFilter(MyFilter(rank))

    def set_log_to_stream(self, level=logging.DEBUG):
        chdr = logging.StreamHandler(sys.stdout)
        chdr.setLevel(level)
        chdr.setFormatter(self.formatter)
        self.addHandler(chdr)

    def set_log_to_file(self, log_file, level=logging.DEBUG):
        log_path = Path(log_file).resolve()
        Path.mkdir(log_path.parent, parents=True, exist_ok=True)
        fhdr = logging.FileHandler(log_path)
        fhdr.setLevel(level)
        fhdr.setFormatter(self.formatter)
        self.addHandler(fhdr)

    def set_log_to_slack(self, credential_file, ch_name, level=logging.INFO):
        try:
            credential_path = Path(credential_file).resolve()
            shdr = SlackClientHandler(credential_path, ch_name)
            shdr.setLevel(level)
            shdr.setFormatter(self.formatter)
            self.addHandler(shdr)
        except:
            raise RuntimeError


logging.setLoggerClass(MyLogger)
logger = logging.getLogger("pytorch-cxr")
logger.setLevel(logging.DEBUG)
logger.propagate = False


def print_versions():
    logger.info(f"pytorch version: {torch.__version__}")
    logger.info(f"torchvision version: {torchvision.__version__}")


def get_devices(cuda=None):
    if cuda is None:
        logger.info(f"use CPUs")
        return [torch.device("cpu")]
    else:
        assert torch.cuda.is_available()
        avail_devices = list(range(torch.cuda.device_count()))
        use_devices = [int(i) for i in cuda.split(",")]
        assert max(use_devices) in avail_devices
        logger.info(f"use cuda on GPU {use_devices}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in use_devices])
        return [torch.device(f"cuda:{k}") for k in use_devices]

def get_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def get_commit(ignore=False):
    import git
    repo = git.Repo(search_parent_directories=True)
    if ignore:
        logger.warning("git repo dirty check ignored")
    else:
        assert not repo.is_dirty(), "current repository has some changes. please make a commit to run"


    try:
        branch = repo.head.ref.name
    except TypeError:
        branch = "(detached)"
    sha = repo.head.commit.hexsha
    dttm = repo.head.commit.committed_datetime
    return f"{branch}/{sha} ({dttm})"

