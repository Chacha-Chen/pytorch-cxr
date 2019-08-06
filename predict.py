from pathlib import Path

import torch

from utils import logger, print_versions
from dataset import get_study, get_image
from model import Network
#from danet import Network


class PredictEnvironment:

    def __init__(self, out_dim, device, model_file=None):
        self.device = device
        self.model = Network(out_dim, mode="per_study").to(self.device)
        self.thresholds = np.zeros(out_dim)
        if model_file is not None:
            self.load_model(model_file)

    def load_model(self, filename):
        filepath = Path(filename).resolve()
        logger.debug(f"loading the model from {filepath}")
        pkg = torch.load(filepath, map_location=self.device)
        state = pkg['state']
        self.thresholds = pkg['thresholds']

        try:
            self.model.load_state_dict(state, strict=True)
        except:
            # remove 'module.' from keys due to DDP
            new_state = { k.replace('module.', ''): v for k, v in state.items() }
            self.model.load_state_dict(new_state, strict=True)


class Predictor:

    def __init__(self, env):
        self.env = env

    def predict_study(self, study_dir):
        img_files = ["*.png", "*.jpg", "*.PNG", "*.JPG"]
        img_paths = [list(Path(study_dir).rglob(f)) for f in img_files]
        img_paths = [p for l in img_paths for p in l]
        image_tensor = get_study(img_paths).unsqueeze(dim=0)
        return self.predict(image_tensor)

    def predict_image(self, img_file):
        image_tensor = get_image(img_file).unsqueeze(dim=0)
        return self.predict(image_tensor)

    def predict(self, data):
        self.env.model.eval()
        with torch.no_grad():
            data = data.to(self.env.device)
            output = self.env.model(data)
        return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CXR Prediction")
    # for testing
    parser.add_argument('--cuda', default=False, action='store_true', help="use GPU")
    parser.add_argument('--model', default=None, type=str, help="pretrained model to be used in prediction")
    #parser.add_argument('study_dirs', type=str, nargs='+', help="list of study directories for prediction")
    args = parser.parse_args()

    if args.cuda:
        assert torch.cuda.is_available()
        device = "cuda"
    else:
        device = "cpu"

    print_versions()

    #model_path = Path(args.model).resolve()
    model_path = Path("train_20190526_per_study/model_epoch_030.pth.tar").resolve()

    env = PredictEnvironment(14, device, model_path)
    p = Predictor(env)

    study_dirs = ["/mnt/hdd/cxr/Stanford/full_resolution_version/CheXpert-v1.0/valid/patient64541/study1"]
    for s in study_dirs:
        logger.info(f"predict the study: {s}")
        out = p.predict_study(s)
        out = torch.sigmoid(out.squeeze())
        vec = " ".join([f"{k:.6f}" for k in out.numpy()])
        logger.info(f"output: {vec}")

