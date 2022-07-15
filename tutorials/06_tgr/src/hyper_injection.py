import numpy as np
import os
import yaml
import pickle
import csv
import matplotlib
import h5py
import json

import torch
from torch.utils.data import DataLoader
import numpy as np  # 1.19.0
import matplotlib.ticker as ticker
import torchvision
from scipy import stats
import matplotlib.patches as mpatches
from functools import partial
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy
from tqdm import tqdm_notebook as tqdm
import pystan
import bilby
import pandas as pd
from pprint import pprint

# os.environ['MPLCONFIGDIR'] = '/home/local/nihargupte'
import dingo.gw.dataset.generate_dataset
from dingo.gw.dataset import WaveformDataset
import dingo.gw.training.train_builders
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
import dingo.gw.waveform_generator
from dingo.gw.domains import build_domain, build_domain_from_model_metadata
import dingo.gw.inference
from dingo.gw.gwutils import get_window_factor
from dingo.gw.transforms import *
from dingo.core.models import PosteriorModel
from dingo.gw.inference import injection
from dingo.gw.inference.gw_samplers import GWSamplerGNPE, GWSampler

# Code developed by Max Isi:
def fit_it(chi_samples, nobs, nsamp, model, niter=2000):
    """Carry out Stan population fit for a given parameter ("chi").

    Arguments
    ---------
    chi_samples: array
        samples for all events `[[event_1_samples], [event_2_samples] ...]`
    nobs: int
        number of events (i.e. observations).
    nsamp: int
        number of samples per event (subselects for speed).
    model: pystan.StanModel
        stan model.
    niter: int
        number of stan iterations (def. 2000).
    """
    # samples = [cs[:nsamp] for cs in chi_samples[:nobs]]
    chosen_chi_samples = chi_samples[
        np.random.choice(range(len(chi_samples)), nobs, replace=False)
    ]
    samples = []
    for cs in chosen_chi_samples:
        idxs = np.random.choice(range(len(cs)), nsamp, replace=False)
        samples.append(cs[idxs])
    stan_data = {"nobs": nobs, "nsamp": nsamp, "chis": samples}
    return model.sampling(data=stan_data, iter=niter)


def save_cache(fits, cache_path, paths=None):
    for param, fit in fits.items():
        try:
            post = fit.extract(["mu", "sigma", "pop"])
        except AttributeError:
            post = fit
        for k in ["mu", "sigma", "pop"]:
            np.savetxt(cache_path.format(par=param, hyper=k), post[k])
        print("Cached %s" % param)
    # record list of events
    if paths:
        cache_dir = os.path.dirname(cache_path)
        log_path = os.path.join(cache_dir, "events.json")
        with open(log_path, "w") as f:
            json.dump(paths, f, indent=4)
        print("Events logged: %r" % log_path)


with open(
    "/home/local/nihargupte/dingo-devel/tutorials/06_tgr/dumps/pystan/posteriors.pkl",
    "rb",
) as f:
    posteriors = pickle.load(f)

model = pystan.StanModel(
    file="/home/local/nihargupte/dingo-devel/tutorials/06_tgr/src/hier_gr.stan"
)
cache_path = "/home/local/nihargupte/dingo-devel/tutorials/06_tgr/dumps/pystan"
# Fit it! (only if needed: `fit_all` checks whether fit already exists)
# data ?= {"domega220":np.array[[event_1_samples], [event_2_samples] ...], "dtau220":[[samples1], [samples2]]}
# save_cache(fits, cache_path)
# print("begin")
fits_dtau220 = fit_it(posteriors["dtau220"], 5, 50_000, model)
fits_domega220 = fit_it(posteriors["domega220"], 5, 50_000, model)
fits = {"model": model, "dtau220": fits_dtau220, "domega220": fits_domega220}
with open("/home/local/nihargupte/dingo-devel/tutorials/06_tgr/dumps/pystan/fits.pkl", "wb") as f:
    pickle.dump(fits, f)
