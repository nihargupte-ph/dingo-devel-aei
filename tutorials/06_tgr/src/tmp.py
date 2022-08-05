import os
import yaml
import pickle
import matplotlib
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union
from numbers import Number

import torch
import numpy as np
import matplotlib.pyplot as plt
import bilby
import torch
import pandas as pd
import pycbc.psd
from scipy.signal import tukey
from pprint import pprint
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters,
    bilby_to_lalsimulation_spins,
)
from pprint import pprint
import lalinference.imrtgr.nrutils as nrutils
import lal
from gwpy.timeseries import TimeSeries
import scipy

import lalsimulation as LS


matplotlib.use('pdf')
os.environ['MPLCONFIGDIR'] = '/home/local/nihargupte'
import dingo.gw.dataset.generate_dataset 
from dingo.gw.inference import injection
import dingo.gw.training.train_builders
import dingo.gw.waveform_generator
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset
from dingo.gw.dataset import WaveformDataset
from dingo.gw.domains import build_domain, build_domain_from_model_metadata
import dingo.gw.inference
from dingo.gw.transforms import SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, UnpackDict, SampleExtrinsicParameters, GNPECoalescenceTimes, AddWhiteNoiseComplex
from dingo.core.models import PosteriorModel
import dingo.gw.domains
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.inference.gw_samplers import GWSamplerGNPE, GWSampler

def _convert_to_scalar(x: Union[np.ndarray, float]) -> Number:
    """
    Convert a single element array to a number.

    Parameters
    ----------
    x:
        Array or number

    Returns
    -------
    A number
    """
    if isinstance(x, np.ndarray):
        if x.shape == () or x.shape == (1,):
            return x.item()
        else:
            raise ValueError(
                f"Expected an array of length one, but shape = {x.shape}"
            )
    else:
        return x


def convert_parameters_to_lal_frame(parameter_dict: Dict, approximant, domain_dict, lal_params=None) -> Tuple:
    """Convert to lal source frame parameters

    Parameters
    ----------
    parameter_dict : Dict
        A dictionary of parameter names and 1-dimensional prior distribution
        objects. If None, we use a default binary black hole prior.
    lal_params : (None, or Swig Object of type 'tagLALDict *')
        Extra parameters which can be passed to lalsimulation calls.

    Returns
    -------
    lal_parameter_tuple:
        A tuple of parameters for the lalsimulation waveform generator
    """
    # Transform mass, spin, and distance parameters
    p, _ = convert_to_lal_binary_black_hole_parameters(parameter_dict)

    # Convert to SI units
    p["mass_1"] *= lal.MSUN_SI
    p["mass_2"] *= lal.MSUN_SI
    p["luminosity_distance"] *= 1e6 * lal.PC_SI

    # Transform to lal source frame: iota and Cartesian spin components
    param_keys_in = (
        "theta_jn",
        "phi_jl",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "a_1",
        "a_2",
        "mass_1",
        "mass_2",
        "f_ref",
        "phase",
    )
    param_values_in = [p[k] for k in param_keys_in]
    iota_and_cart_spins = bilby_to_lalsimulation_spins(*param_values_in)
    iota, s1x, s1y, s1z, s2x, s2y, s2z = [
        float(_convert_to_scalar(x)) for x in iota_and_cart_spins
    ]

    # Construct argument list for FD and TD lal waveform generator wrappers
    spins_cartesian = s1x, s1y, s1z, s2x, s2y, s2z
    masses = (p["mass_1"], p["mass_2"])
    extra_params = (p["luminosity_distance"], iota, p["phase"])
    ecc_params = (0.0, 0.0, 0.0)  # longAscNodes, eccentricity, meanPerAno

    if "delta_t" in domain_dict.keys():
        domain_pars = (domain_dict["delta_t"], domain_dict["f_min"], domain_dict["f_ref"])
    elif "delta_f" in domain_dict.keys():
        domain_pars = ((domain_dict["delta_f"], domain_dict["f_start"], domain_dict["f_max"], domain_dict["f_ref"]))

    lal_dict = lal.CreateDict()
    lal_parameter_tuple = (
        domain_pars[0],
        *masses,
        domain_pars[1],
        1e6 * lal.PC_SI, # Distance
        s1z,
        s2z,
        4111,
        0, 
        0, 
        0, 
        0,
        0, 
        0,
        0, 
        0, 
        0,
        0,
        parameters["domega220"],
        parameters["dtau220"],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        None, 
        0,
        lal_dict
    )
    return lal_parameter_tuple


parameters = {
    "phase": 0.,
    "mass_1": 38.9, # Solar Masses
    "mass_2": 31.6, # Solar Masses
    "f_ref": 20., # Hz
    "phase": 0.,
    "chi_1": 0.,
    "chi_2": 0.,
    "theta_jn": 0,
    "luminosity_distance": 410., #Mpc
    "domega220": 0.,
    "dtau220": 0.
}

domain_dict = {
    "f_min": 20,
    "f_ref": 20,
    "f_max": 2048
}

time_duration = 8.0
sampling_rate = 2 * 2 * domain_dict["f_max"]
domain_dict["delta_t"] = 1 / sampling_rate


# REAL8 deltaT, const REAL8 m1SI, const REAL8 m2SI, const REAL8 fMin, const REAL8 r, const REAL8 spin1z, const REAL8 spin2z, 
# UINT4 SpinAlignedEOBversion, const REAL8 lambda2Tidal1, const REAL8 lambda2Tidal2, const REAL8 omega02Tidal1, const REAL8 omega02Tidal2, 
# const REAL8 lambda3Tidal1, const REAL8 lambda3Tidal2, const REAL8 omega03Tidal1, const REAL8 omega03Tidal2, const REAL8 quadparam1, 
# const REAL8 quadparam2, const REAL8 domega220, const REAL8 dtau220, const REAL8 domega210, const REAL8 dtau210, const REAL8 domega330, const REAL8 dtau330, 
# const REAL8 domega440, const REAL8 dtau440, const REAL8 domega550, const REAL8 dtau550, REAL8Vector *nqcCoeffsInput, const INT4 nqcFlag, LALDict *PAparams)
lal_params = convert_parameters_to_lal_frame(parameters, LS.SEOBNRv4HM_PA, domain_dict, None)
print(lal_params)
LS.SimIMRSpinAlignedEOBModes(*lal_params)