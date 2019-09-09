from pathlib import Path

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import logger, get_devices, print_versions
from dataset import STANFORD_CXR_BASE, MIMIC_CXR_BASE, NIH_CXR_BASE, MODES, get_study, get_image, cxr_test_transforms
from model import Network
#from danet import Network

pd.set_option('mode.chained_assignment', None)


class PredictEnvironment:

    def __init__(self, out_dim, device, mode="per_study", model_path=None):
        self.device = device
        self.mode = mode
        self.model = Network(out_dim, mode=mode).to(self.device)
        self.thresholds = np.zeros(out_dim)
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, filename):
        filepath = Path(filename).resolve()
        logger.debug(f"loading the model from {filepath}")

        ckpt = torch.load(filepath, map_location=self.device)

        if 'model_state' in ckpt:
            model_state = ckpt['model_state']
        elif 'state' in ckpt:   # for legacy
            model_state = ckpt['state']
        else:                   # for legacy
            model_state = ckpt
        try:
            self.model.load_state_dict(model_state, strict=True)
        except:
            # remove 'module.' from keys due to DDP
            new_state = { k.replace('module.', ''): v for k, v in model_state.items() }
            self.model.load_state_dict(new_state, strict=True)
        if 'thresholds' in ckpt:
            self.thresholds = ckpt['thresholds']

        return ckpt


class Predictor:

    def __init__(self, env):
        self.env = env

    def get_study_input(self, study_dir):
        img_files = ["*.png", "*.jpg", "*.PNG", "*.JPG"]
        img_paths = [list(Path(study_dir).rglob(f)) for f in img_files]
        img_paths = [p for l in img_paths for p in l]
        image_tensor = get_study(img_paths, None, cxr_test_transforms, use_memcache=False)
        return image_tensor

    def get_image_input(self, img_file):
        image_tensor = get_image(img_file, cxr_test_transforms, use_memcache=False)
        return image_tensor

    def predict(self, input_path):
        if self.env.mode == "per_study":
            x = self.get_study_input(input_path)
        else:
            x = self.get_image_input(input_path)

        #img = x[0, :, :].squeeze()
        #plt.imshow(img.squeeze(), cmap="gray", interpolation='none')
        #plt.show()

        x = x.unsqueeze(dim=0)  # to make 1-batched input tensor

        self.env.model.eval()
        with torch.no_grad():
            x = x.to(self.env.device)
            output = self.env.model(x)
        return output


def load_manifest(file_path, mode="per_study"):
    assert mode in MODES
    if not file_path.exists():
        logger.error(f"manifest file {file_path} not found.")
        sys.exit(1)

    logger.debug(f"loading dataset manifest {file_path} ...")
    df = pd.read_csv(str(file_path)).fillna(0)
    df_tmp = df[[df.columns[0]]]
    if mode == "per_image":
        entries = df_tmp
    elif mode == "per_study":
        logger.debug("grouping by studies ... ")
        df_tmp['study'] = df_tmp.apply(lambda x: str(Path(x[0]).parent), axis=1)
        df_tmp.set_index(['study'], inplace=True)
        aggs = { df_tmp.columns[0]: lambda x: ','.join(x.astype(str)) }
        df_tmp = df_tmp.groupby(['study']).agg(aggs).reset_index(0, drop=True)
        entries = df_tmp
    else:
        raise RuntimeError

    logger.debug(f"{len(entries)} entries are loaded.")
    return entries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CXR Prediction")
    # for testing
    parser.add_argument('--cuda', default=None, type=str, help="use GPUs with its device ids, separated by commas")
    parser.add_argument('--model', default=None, type=str, help="pretrained model to be used in prediction")
    parser.add_argument('input_csv', type=str, help="input csv filepath for test")
    parser.add_argument('output_csv', type=str, help="output csv filepath for test")
    args = parser.parse_args()

    device = get_devices(args.cuda)[0]
    print_versions()

    mode = "per_study"
    LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    model_path = Path(args.model).resolve()
    #model_path = Path("train_20190526_per_study/model_epoch_030.pth.tar").resolve()
    env = PredictEnvironment(5, device, mode, model_path=model_path)
    p = Predictor(env)

    input_path = Path(args.input_csv).resolve()
    entries = load_manifest(input_path, mode)
    base_path = input_path.parent
    outputs = { "Study": [], }
    outputs.update({ k: [] for k in LABELS})
    for i, r in entries.iterrows():
        paths = [base_path.joinpath(x).resolve() for x in r[0].split(',')]
        study_path = paths[0].parent
        out = p.predict(study_path)
        out = torch.sigmoid(out.squeeze())
        outputs["Study"].append(study_path)
        for j, l in enumerate(LABELS):
            outputs[l].append(out[j].cpu().item())
        vec = " ".join([f"{k:.6f}" for k in out.cpu().numpy()])
        logger.info(f"predict {i:03d}: {study_path} {vec}")

    output_df = pd.DataFrame(outputs)
    print(output_df)
    output_df.to_csv(args.output_csv, index=False)
