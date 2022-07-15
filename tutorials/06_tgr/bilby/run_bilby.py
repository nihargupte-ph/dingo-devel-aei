#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import yaml
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
from bilby.gw.detector import PowerSpectralDensity
from bilby.gw.detector import InterferometerList

matplotlib.use("pdf")

os.environ["MPLCONFIGDIR"] = "/home/local/nihargupte"
import dingo.gw.dataset.generate_dataset
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
import dingo.gw.training.train_builders
import dingo.gw.waveform_generator
from dingo.gw.domains import build_domain, build_domain_from_model_metadata
import dingo.gw.inference
from dingo.gw.transforms import *
from dingo.gw.gwutils import get_window_factor
from dingo.gw.inference import injection
from dingo.core.models import PosteriorModel


# In[5]:
setproctitle.setproctitle(f"bilby_dingo")
approximant = "SEOBNRv4HM_PA"
"/home/local/nihargupte/dingo-devel/tutorials/06_tgr/bilby"
train_dir = "/home/local/nihargupte/dingo-devel/tutorials/06_tgr/train_dir_SEOBNRv4HM_PA_O1_2048"
save_dir = f"/home/local/nihargupte/dingo-devel/tutorials/06_tgr/bilby/injection_1"
os.makedirs(save_dir, exist_ok=True)

torch.cuda.set_device(4)
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
asd_dataset = ASDDataset(file_name= "/home/local/nihargupte/dingo-devel/tutorials/03_aligned_spin/datasets/ASDs_new/1024_1/asds_O1.hdf5")
asd = asd_dataset.sample_random_asds()
injection_generator.asd = asd
injection_generator.whiten = False

# GW150914 intrinsic mass
intrinsic_parameters = {
    # intrinsic parameters
    "mass_1": 39.4,
    "mass_2": 30.9,
    "chi_1": 0.32,
    "chi_2": 0.57,
    "domega220": 0.2,
    "dtau220": 0.3,
}

extrinsic_parameters = {
    "phase": 0,  # rad
    "theta_jn": 2.6,  # rad inclination to maximize effect of higher modes
    "psi": 0,  # rad
    "ra": 0,  # rad
    "dec": 0,  # rad
    "geocent_time": 0.0,  # s
    "luminosity_distance": 390,  # Mpc
}
theta = {**intrinsic_parameters, **extrinsic_parameters}
chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(theta["mass_1"], theta["mass_2"])
mass_ratio = bilby.gw.conversion.component_masses_to_mass_ratio(theta["mass_1"], theta["mass_2"])
theta["mass_ratio"] = mass_ratio
theta["chirp_mass"] = chirp_mass
print(theta)

domain = build_domain_from_model_metadata(metadata)
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated())
strain_data = injection_generator.injection(theta)

strain_data["parameters"] = theta
with open(save_dir + "/strain_data.pkl", "wb") as f:
    pickle.dump(strain_data, f)

# In[8]:
ifo_strs = ["H1", "L1"]
ifos = InterferometerList(ifo_strs)
for ifo in ifos:
    asd_array = strain_data["asds"][ifo.name]
    # This step is actually very important
    asd_array = asd_array.astype("float64")
    x_freq = domain.sample_frequencies.astype("float64")
    psd = PowerSpectralDensity(frequency_array=x_freq, asd_array=asd_array)
    ifo.power_spectral_density = psd
    # Making sure we aren't dropping anything due to precision
    assert np.min(asd_array**2) > 0
    ifo.set_strain_data_from_frequency_domain_strain(strain_data["waveform"][ifo.name], frequency_array=x_freq)
    ifo.strain_data.window_factor = domain.window_factor
    # NOTE SET WINDOW FACTOR HERE, bilby stores asd in multiple places so be careful. Sometimes asd is asd*sqrt(window_factor) != asd

# %%
# Setting the Priors
priors = injection_generator.prior

priors["geocent_time"].minimum = metadata["train_settings"]["data"]["ref_time"] - 0.1
priors["geocent_time"].maximum = metadata["train_settings"]["data"]["ref_time"] + 0.1

# Making sure that the injection lies within the prior (particuarly for the mass)
assert (priors["mass_ratio"].minimum < theta["mass_ratio"]and theta["mass_ratio"] < priors["mass_ratio"].maximum)
assert (priors["chirp_mass"].minimum < theta["chirp_mass"]and theta["chirp_mass"] < priors["chirp_mass"].maximum)

# Specify the output directory and the name of the simulation.
outdir = f"{save_dir}/outdir"
bilby.core.utils.setup_logger(outdir=outdir, label=approximant)

# Creating the Waveform Generator
# specify waveform arguments
waveform_arguments = dict(
    waveform_approximant=approximant,  # waveform approximant name
    reference_frequency=20.0,  # gravitational waveform reference frequency (Hz)
)
time_duration = 8.0  # time duration (seconds)
sampling_frequency = 8192.0  # sampling frequency (Hz) NOTE set to 4096.0 if f_max=1024.0

waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=sampling_frequency,
    duration=time_duration,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    parameters=strain_data["parameters"],
    waveform_arguments=waveform_arguments,
)

# compute the likelihoods
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=waveform_generator,
    priors=priors,
)

result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="bilby_mcmc",
    nsamples=100,
    outdir=outdir,
    npool=16,
    )
