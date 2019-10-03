#!/bin/env/python
import subprocess as sp
from pathlib import Path

base_dir = Path("runtime").resolve()

title = "cxr_single"

#20190922_noniid_max_single_stanford_per_image_custom
#20190923_noniid_max_single_stanford_per_image_postive_weight
#20190924_noniid_max_dist_per_image_no_postive_weight
#20190924_noniid_max_single_stanford_per_study_no_positive_weight
#20190925_noniid_max_single_stanford_per_study_positive_weight
#20190927_noniid_max_dist_per_study_no_postive_weight
#20190927_noniid_max_single_stanford_per_study_320
#20190927_noniid_max_single_stanford_per_study_512
#20190927_noniid_max_single_stanford_per_study_no_positive_weight_dropout
#20190927_noniid_max_single_stanford_per_study_no_positive_weight_dropout_radam

plots = {
    "single_per_image_nopw": "20190922_noniid_max_single_stanford_per_image_custom/tensorboard.0/",
    "single_per_image_pw": "20190923_noniid_max_single_stanford_per_image_postive_weight/tensorboard.0/",
    "single_per_study_nopw": "20190924_noniid_max_single_stanford_per_study_no_positive_weight/tensorboard.0/",
    "single_per_study_pw": "20190925_noniid_max_single_stanford_per_study_positive_weight/tensorboard.0/",
    "single_per_study_320": "20190927_noniid_max_single_stanford_per_study_320/tensorboard.0/",
    "single_per_study_densenet169": "20190929_noniid_max_single_stanford_per_study_densenet169/tensorboard.0/",
    "single_per_study_densenet161": "20190930_noniid_max_single_stanford_per_study_densenet161/tensorboard.0/",
    "dist_per_image_nopw": "20190924_noniid_max_dist_per_image_no_postive_weight/tensorboard.0/",
    "dist_per_study_densenet169": "20190929_noniid_max_dist_per_study_densenet169/tensorboard.0/",
    "single_per_study_densenet169_dropout": "20191001_noniid_max_single_stanford_per_study_densenet169_dropout/tensorboard.0/",
}

logdir = ",".join([f"{k}:{base_dir.joinpath(v)}" for k, v in plots.items()])

cmd = ["tensorboard", "--host", "0.0.0.0", "--window_title", title, "--logdir", logdir]
print(" ".join(cmd))
sp.run(cmd)
