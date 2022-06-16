#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import yaml
import gc
import pickle
import matplotlib

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.ticker as ticker
from scipy import stats
import matplotlib.pyplot as plt
import bilby
import setproctitle
import pandas as pd
from threadpoolctl import threadpool_limits
from pprint import pprint

matplotlib.use("pdf")

os.environ["MPLCONFIGDIR"] = "/home/local/nihargupte"
import dingo.gw.dataset.generate_dataset
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
import dingo.gw.training.train_builders
import dingo.gw.waveform_generator
from dingo.gw.domains import build_domain, build_domain_from_model_metadata
import dingo.gw.inference
from dingo.gw.transforms import *
from dingo.gw.inference import injection
from dingo.gw.gwutils import get_window_factor
from dingo.core.models import PosteriorModel


# In[5]:
setproctitle.setproctitle(f"bilby_dingo")
approximant = "SEOBNRv4HM_ROM"
special = "_O1_2048"
save_idx = 22
models_dir = "/home/local/nihargupte/dingo-devel/tutorials/03_aligned_spin"
train_dir = models_dir + f"/train_dir_{approximant}{special}"
save_dir = f"{approximant}{special}_comparison/{save_idx}"
waveform_generation_dir = (
    "/home/local/nihargupte/dingo-devel/tutorials/03_aligned_spin/datasets/waveforms"
)
os.makedirs(save_dir, exist_ok=True)
os.environ[
    "LAL_DATA_PATH"
] = "/home/local/nihargupte/dingo-devel/venv/lib/python3.9/site-packages/lalsimulation/"

torch.cuda.set_device(0)
if not os.path.exists(train_dir + "/metadata.yaml"):
    main_pm = PosteriorModel(
        device="cuda",
        **{"model_filename": f"{train_dir}/model_latest.pt"},
        load_training_info=False,
    )
    with open(train_dir + "/metadata.yaml", "w") as f:
        yaml.safe_dump(main_pm.metadata, f)

    exit()

with open(train_dir + "/metadata.yaml", "r") as f:
    metadata = yaml.safe_load(f)

injection_generator = injection.Injection.from_posterior_model_metadata(metadata)

# Opening up a asd
# NOTE HARD CODING WHERE THE ASD IS STORED SINCE IT WAS MOVED
metadata["train_settings"]["training"]["stage_0"][
    "asd_dataset_path"
] = "/home/local/nihargupte/dingo-devel/tutorials/03_aligned_spin/datasets/ASDs_new/1024_1/asds_O1.hdf5"
asd_dataset = ASDDataset(
    file_name=metadata["train_settings"]["training"]["stage_0"][
        "asd_dataset_path"
    ]
)
asd = asd_dataset.sample_random_asds()
injection_generator.asd = asd
injection_generator.whiten = False

# GW150914 intrinsic mass
intrinsic_parameters = {
    # intrinsic parameters
    "chirp_mass": 22,
    "mass_ratio": 0.5,
    "chi_1": 0.5,
    "chi_2": 0.3,
}

extrinsic_parameters = {
    "phase": 1.2,  # rad
    "theta_jn": 0,  # rad inclination to maximize effect of higher modes
    "psi": 0.7,  # rad
    "ra": 0,  # rad
    "dec": 0,  # rad
    "geocent_time": 0.0,  # s
    "luminosity_distance": 410,  # Mpc
}

theta = {**intrinsic_parameters, **extrinsic_parameters}
domain = build_domain_from_model_metadata(metadata)
strain_data = injection_generator.injection(theta)

strain_data["parameters"] = theta
with open(save_dir + "/strain_data.pkl", "wb") as f:
    pickle.dump(strain_data, f)

# In[8]:
from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.detector import InterferometerList

os.environ[
    "LAL_DATA_PATH"
] = "/home/local/nihargupte/dingo-devel/venv/lib/python3.9/site-packages/lalsimulation/"

ifo_strs = ["H1", "L1"]
ifo_list = InterferometerList(ifo_strs)
for ifo in ifo_list:
    asd_array = strain_data["asds"][ifo.name]
    # This step is actually very important
    asd_array = asd_array.astype("float64")
    x_freq = domain.sample_frequencies.astype("float64")
    psd = PowerSpectralDensity(frequency_array=x_freq, asd_array=asd_array)
    ifo.power_spectral_density = psd
    # Making sure we aren't dropping anything due to precision
    assert np.min(asd_array**2) > 0
    ifo.set_strain_data_from_frequency_domain_strain(strain_data["waveform"][ifo.name], frequency_array=x_freq)
    ifo.strain_data.roll_off = metadata["train_settings"]["data"]["window"]["roll_off"] # Set this explicitly. Default is 0.2.

    # NOTE SET WINDOW FACTOR HERE, bilby stores asd in multiple places so be careful. Sometimes asd is asd*sqrt(window_factor) != asd

# %%
# Setting the Priors
priors = injection_generator.prior

priors["geocent_time"].minimum = strain_data["parameters"]["geocent_time"] - 0.1
priors["geocent_time"].maximum = strain_data["parameters"]["geocent_time"] + 0.1

# Making sure that the injection lies within the prior (particuarly for the mass)
assert (
    priors["mass_ratio"].minimum < theta["mass_ratio"]
    and theta["mass_ratio"] < priors["mass_ratio"].maximum
)
assert (
    priors["chirp_mass"].minimum < theta["chirp_mass"]
    and theta["chirp_mass"] < priors["chirp_mass"].maximum
)

# Specify the output directory and the name of the simulation.
outdir = f"{save_dir}/outdir"
bilby.core.utils.setup_logger(outdir=outdir, label=approximant)

# NOTE Pass reference time to Bilby
time_duration = 8.0  # time duration (seconds)

# In this step we define a `waveform_generator`. This is the object which
# creates the frequency-domain strain. In this instance, we are using the
# `lal_binary_black_hole model` source model. We also pass other parameters:
# the waveform approximant and reference frequency and a parameter conversion
# which allows us to sample in chirp mass and ratio rather than component mass
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={'waveform_approximant': approximant,
                        'reference_frequency': 20,
                        'minimum_frequency': 20,
                        'maximum_frequency': 2048})

# In this step, we define the likelihood. Here we use the standard likelihood
# function, passing it the data and the waveform generator.
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(ifo_list, waveform_generator, priors=priors)

# Finally, we run the sampler. This function takes the likelihood and prior
# along with some options for how to do the sampling and how to save the data
result = bilby.run_sampler(
    likelihood, priors, sampler='dynesty', outdir=save_dir, label=save_idx,
    nlive=2000, nact=10, walks=100, n_check_point=10000, check_point_plot=True,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    plot=False)
