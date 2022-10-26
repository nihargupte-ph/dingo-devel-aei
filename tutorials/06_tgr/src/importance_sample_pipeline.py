import numpy as np
import os
import h5py

import torch
import numpy as np # 1.19.0
import pandas as pd

from dingo.gw.inference.data_preparation import get_event_data_and_domain
from dingo.core.models import PosteriorModel
from dingo.core.samples_dataset import SamplesDataset
from dingo.gw.importance_sampling.importance_weights import importance_sample
from dingo.gw.inference.gw_samplers import GWSamplerGNPE, GWSampler

torch.cuda.set_device(3)

# GW200115_042309 strange error

# build posterior model
# For O1 you want to use model_stage_1
main_pm = PosteriorModel(
    device="cuda",
    **{"model_filename": "/data/nihargupte/projects/dingo-devel/tutorials/06_tgr/train_dir_SEOBNRv4HM_PA_O3_1600_HLV/model_latest.pt"},
    load_training_info=False
)

time_pm = PosteriorModel(
    device='cuda',
    **{"model_filename": "/data/nihargupte/projects/dingo-devel/tutorials/06_tgr/train_dir_SEOBNRv4HM_PA_O3_1600_HLV_time/model_latest.pt"},
    load_training_info=False
)


# Downloading data
init_sampler = GWSampler(model=time_pm)
sampler = GWSamplerGNPE(model=main_pm, init_sampler=init_sampler, num_iterations=40)

time_psd = 1024
time_buffer = 2.0
time_event = 1251009263.7
event_data, domain = get_event_data_and_domain(main_pm.metadata, time_event, time_psd, time_buffer)
sampler.context = event_data

sampler.event_metadata = {
    "time_event": time_event,
    "time_psd": time_psd,
    "time_buffer": time_buffer,
}

# Sampling Points
sampler.context = event_data
sampler.run_sampler(num_samples=1_500_000, batch_size=5_000)
sampler.samples = sampler.samples.drop(columns=[param for param in sampler.samples.columns  if "GNPE:" in param])


# Saving samples 
# Save training samples
sampler.samples = sampler.samples.drop(columns=[param for param in sampler.samples.columns  if "GNPE:" in param or "phase"==param])
# Throw away phase because of synthetic phase NOTE should we could also marginalize over this but 
# sampler.metadata["train_settings"]["data"]["inference_parameters"].remove("phase")
sampler.base_model_metadata["dataset_settings"]["waveform_generator"]["spin_conversion_phase"] = 0 
outdir = f"/data/nihargupte/projects/dingo-devel/tutorials/06_tgr/dumps/2_detector_O3/GW190828_063405/is"
os.makedirs(outdir)
sampler.to_hdf5(label="training_samples", outdir=outdir)

# Load training samples
# sampler.metadata["train_settings"]["data"]["inference_parameters"].remove("phase")
sampler.base_model_metadata["dataset_settings"]["waveform_generator"]["spin_conversion_phase"] = 0 
labels = main_pm.metadata["train_settings"]["data"]["inference_parameters"].copy()
with h5py.File(f"{outdir}/dingo_samples_training_samples.hdf5", "r") as f:
    a = f["samples"].fields(labels)[:]
    arr = np.empty((a.shape[0], len(labels)))
    for i in range(a.shape[0]):
        arr[i, :] = list(a[i])

sampler.samples = pd.DataFrame(data=arr, columns=labels)



# Importance Sampling

sampler.metadata["event"] = {}
sampler.metadata["event"]["time_event"] = time_event
samples_dataset = SamplesDataset(
    dictionary={
        "samples": sampler.samples,
        "context": event_data,
        "settings": sampler.metadata,
    }
)
settings = {
    "num_samples": 250_000,
    "num_processes": 16,
    # "slice_plots": {
    #     "num_slice_plots": 5,
    #     "params_slice2d": [["phase", "geocent_time"], ["phase", "chi_1"]],
    # },
    "synthetic_phase": {
        "approximation_22_mode": False,
        "n_grid": 5001,
        "uniform_weight": 0.01,
        "num_processes": 16,
    },
    "nde": {
        "data": {"parameters": list(main_pm.metadata["train_settings"]["data"]["inference_parameters"].copy())},
        "model": {
            "type": "nsf",
            "num_flow_steps": 8,
            "base_transform_kwargs": {
                "hidden_dim": 128,
                "num_transform_blocks": 10,
                "activation": "elu",
                "dropout_probability": 0.1,
                "batch_norm": True,
                "num_bins": 8,
                "base_transform_type": "rq-coupling",
            },
        },
        "training": {
            "device": "cuda",
            "num_workers": 0,
            "train_fraction": 0.9,
            "batch_size": 4096,
            "epochs": 20,
            "optimizer": {"type": "adam", "lr": 0.000835},
            "scheduler": {"type": "cosine", "T_max": 20},
        },       
    },
}
importance_sample(settings, samples_dataset, outdir)
