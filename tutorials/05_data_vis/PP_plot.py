import os
import setproctitle
import yaml
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.ticker as ticker
from scipy import stats
import matplotlib.pyplot as plt
import bilby
import csv
import tqdm
import tqdm
import torchvision
import pandas as pd
import pprint

from dingo.core.models import PosteriorModel
from dingo.gw.transforms import (
    SelectStandardizeRepackageParameters,
    RepackageStrainsAndASDS,
    UnpackDict,
)
import dingo
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net
from dingo.api import resubmit_condor_job
from dingo.gw.training import build_dataset, set_train_transforms
from dingo.gw.gwutils import *
from dingo.gw.inference import injection
from dingo.gw.inference.gw_samplers import GWSamplerGNPE, GWSampler

#%%
approximant = "SEOBNRv4HM_ROM"
os.environ["LAL_DATA_PATH"] = "/home/local/nihargupte/dingo-devel/venv/lib/python3.9/site-packages/lalsimulation/"
special = "O1_2048"
train_dir = f"/home/local/nihargupte/dingo-devel/tutorials/03_aligned_spin/train_dir_{approximant}_{special}"
time_train_dir = f"/home/local/nihargupte/dingo-devel/tutorials/03_aligned_spin/train_dir_{approximant}_{special}_time"
save_dir = f"/home/local/nihargupte/dingo-devel/tutorials/05_data_vis/{approximant}_{special}_comparison"
os.makedirs(save_dir, exist_ok=True)
setproctitle.setproctitle(f"py_plot")

torch.cuda.empty_cache()
torch.cuda.set_device(0)

pm_kwargs = {"model_filename": f"{train_dir}/model_latest.pt"}
# build posterior model
main_pm = PosteriorModel(
    device="cuda",
    **{"model_filename": f"{train_dir}/model_latest.pt"},
    load_training_info=False,
)

time_pm = PosteriorModel(
    device="cuda",
    **{"model_filename": f"{time_train_dir}/model_latest.pt"},
    load_training_info=False,
)


injection_generator = injection.Injection.from_posterior_model(main_pm)
# Opening up a asd
asd_dataset = ASDDataset(
    file_name=main_pm.metadata["train_settings"]["training"]["stage_0"][
        "asd_dataset_path"
    ]
)
asd = asd_dataset.sample_random_asds()
injection_generator.asd = asd

np.random.seed(42)
torch.manual_seed(42)
strain_data = injection_generator.random_injection()

init_sampler = GWSampler(model=time_pm)
sampler = GWSamplerGNPE(model=main_pm, init_sampler=init_sampler, num_iterations=30)

def make_pp(percentiles, parameter_labels, ks=True):

    percentiles = percentiles / 100.0
    nparams = percentiles.shape[-1]
    nposteriors = percentiles.shape[0]

    ordered = np.sort(percentiles, axis=0)
    ordered = np.concatenate((np.zeros((1, nparams)), ordered, np.ones((1, nparams))))
    y = np.linspace(0, 1, nposteriors + 2)

    fig = plt.figure(figsize=(10, 10))

    for n in range(nparams):
        if ks:
            pvalue = stats.kstest(percentiles[:, n], "uniform")[1]
            plt.step(
                ordered[:, n],
                y,
                where="post",
                label=parameter_labels[n] + r" ({:.3g})".format(pvalue),
            )
        else:
            plt.step(ordered[:, n], y, where="post", label=parameter_labels[n])
    plt.plot(y, y, "k--")
    plt.legend(prop={"size": 15})
    plt.ylabel(r"$CDF(p)$")
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.xlabel(r"$p$")

    ax = fig.gca()
    ax.set_aspect("equal", anchor="SW")

    plt.grid()
    plt.show()

labels = list(strain_data["parameters"].keys())
labels.remove("H1_time")
labels.remove("L1_time")
neval = 1000  # number of injections
nparams = len(labels)

percentiles = np.empty((neval, nparams))
for idx in range(neval):
    strain_data = injection_generator.random_injection()
    sampler.context = strain_data.copy()
    sampler.run_sampler(
        num_samples=25_000,
        batch_size=25_000,
    )
    pred = np.stack([sampler.samples[l] for l in labels])
    actual = np.array([strain_data["parameters"][l] for l in labels])

    for n in range(nparams):
        percentiles[idx, n] = stats.percentileofscore(pred[n, :], actual[n])

make_pp(percentiles, labels)

with open(f"{save_dir}/PP.pdf", "wb") as f:
    pickle.dump(percentiles, f)
plt.savefig(f"{save_dir}/PP.pdf")
