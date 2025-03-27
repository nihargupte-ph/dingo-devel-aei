import numpy as np
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from bilby.gw.detector import PowerSpectralDensity

from dingo.gw.prior import default_extrinsic_dict, default_intrinsic_dict
from dingo.gw.prior import BBHExtrinsicPriorDict
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_component_masses
from pesummary.gw.conversions import convert
from astropy import cosmology, units


def get_window(window_kwargs):
    """Compute window from window_kwargs."""
    type = window_kwargs["type"]
    if type == "tukey":
        roll_off, T, f_s = (
            window_kwargs["roll_off"],
            window_kwargs["T"],
            window_kwargs["f_s"],
        )
        alpha = 2 * roll_off / T
        w = tukey(int(T * f_s), alpha)
        return w
    else:
        raise NotImplementedError(f"Unknown window type {type}.")


def get_window_factor(window):
    """Compute window factor. If window is not provided as array or tensor but as
    window_kwargs, first build the window."""
    if type(window) == dict:
        window = get_window(window)
    return np.sum(window**2) / len(window)


def get_extrinsic_prior_dict(extrinsic_prior):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default.
    TODO: Move to dingo.gw.prior.py?"""
    extrinsic_prior_dict = default_extrinsic_dict.copy()
    for k, v in extrinsic_prior.items():
        if v.lower() != "default":
            extrinsic_prior_dict[k] = v
    return extrinsic_prior_dict

def get_intrinsic_prior_dict(intrinsic_prior):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default.
    TODO: Move to dingo.gw.prior.py?"""
    intrinsic_prior_dict = default_intrinsic_dict.copy()
    for k, v in intrinsic_prior.items():
        if isinstance(v, int) or isinstance(v, float):
            intrinsic_prior_dict[k] = v
            continue
        if v.lower() != "default":
            intrinsic_prior_dict[k] = v
    return intrinsic_prior_dict


def get_mismatch(a, b, domain, asd_file=None):
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------

    """
    if asd_file is not None:
        # whiten a and b, such that we can use flat-spectrum inner products below
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain.sample_frequencies)
        a = a / asd_array
        b = b / asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((a.conj() * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((a.conj() * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((b.conj() * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


def get_standardization_dict(
    extrinsic_prior_dict, wfd, selected_parameters, transform=None
):
    """
    Calculates the mean and standard deviation of parameters. This is needed for
    standardizing neural-network input and output.

    Parameters
    ----------
    extrinsic_prior_dict : dict
    wfd : WaveformDataset
    selected_parameters : list[str]
        List of parameters for which to estimate standardization factors.
    transform : Transform
        Operator that will generate samples for parameters contained in
        selected_parameters that are not contained in the intrinsic or extrinsic prior.
        (E.g., H1_time, L1_time_proxy)

    Returns
    -------

    """
    # The intrinsic standardization is estimated based on the entire dataset.
    mean_intrinsic, std_intrinsic = wfd.parameter_mean_std()

    # Some of the extrinsic prior parameters have analytic means and standard
    # deviations. If possible, this will either get these, or else it will estimate
    # them numerically.
    ext_prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
    mean_extrinsic, std_extrinsic = ext_prior.mean_std(ext_prior.keys())

    # Check that overlap between intrinsic and extrinsic parameters is only
    # due to fiducial values (-> std 0)
    for k in std_intrinsic.keys() & std_extrinsic.keys():
        assert std_intrinsic[k] == 0

    # Merge dicts, overwriting fiducial values for parameters (e.g.,
    # luminosity_distance) in intrinsic parameters by the extrinsic ones
    mean = {**mean_intrinsic, **mean_extrinsic}
    std = {**std_intrinsic, **std_extrinsic}

    # For all remaining parameters that require standardization, we use the transform
    # to sample these and estimate the mean and standard deviation numerically.
    additional_parameters = [p for p in selected_parameters if p not in mean]
    if additional_parameters:
        num_samples = min(100_000, len(wfd.parameters))
        samples = {p: np.empty(num_samples) for p in additional_parameters}
        for n in range(num_samples):
            sample = {"parameters": wfd.parameters.iloc[n].to_dict()}
            sample = transform(sample)
            for p in additional_parameters:
                # This assumes all of the additional parameters are contained within
                # extrinsic_parameters. We have set it up so this is the case for the
                # GNPE proxies and the detector coalescence times.
                samples[p][n] = sample["extrinsic_parameters"][p]
        mean_additional = {p: np.mean(samples[p]).item() for p in additional_parameters}
        std_additional = {p: np.std(samples[p]).item() for p in additional_parameters}

        mean.update(mean_additional)
        std.update(std_additional)

    standardization_dict = {
        "mean": {k: mean[k] for k in selected_parameters},
        "std": {k: std[k] for k in selected_parameters},
    }
    return standardization_dict


def fill_missing_available_parameters(df):
    """
    Fill in missing parameters using available ones in a gravitational wave parameter dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing samples with missing parameters.
    """
    def chirp_mass(m1, m2):
        """Compute chirp mass given two component masses."""
        return (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)

    # Compute missing total_mass_src and chirp_mass_src if possible
    if "mass_1_src" in df and "mass_ratio" in df:
        df["mass_2_src"] = df.get("mass_2_src", df["mass_1_src"] * df["mass_ratio"])
    if "mass_1_src" in df and "mass_2_src" in df:
        df["total_mass_src"] = df.get("total_mass_src", df["mass_1_src"] + df["mass_2_src"])
        df["chirp_mass_src"] = df.get("chirp_mass_src", chirp_mass(df["mass_1_src"], df["mass_2_src"]))

    # Convert source-frame masses to detector-frame
    for mass_key in ["mass_1_src", "mass_2_src", "total_mass_src", "chirp_mass_src"]:
        if mass_key in df:
            df[mass_key.replace("_src", "")] = df[mass_key] * (1 + df["redshift"])

    # Compute missing mass parameters
    if "mass_1" in df:
        df["mass_2"] = df.get("mass_2", df["mass_1"] * df.get("mass_ratio", np.nan))
        df["mass_ratio"] = df.get("mass_ratio", df["mass_2"] / df["mass_1"])
        if "chirp_mass" not in df and "mass_2" in df:
            df["chirp_mass"] = chirp_mass(df["mass_1"], df["mass_2"])
    elif "chirp_mass" in df and "mass_ratio" in df:
        df["mass_1"], df["mass_2"] = chirp_mass_and_mass_ratio_to_component_masses(df["chirp_mass"], df["mass_ratio"])

    df["total_mass"] = df.get("total_mass", df["mass_1"] + df["mass_2"])

    # Fill missing chi values
    for i in [1, 2]:
        df[f"chi_{i}"] = df.get(f"chi_{i}", df.get(f"a_{i}", np.nan))

    # Compute effective spin parameter
    if "chi_eff" not in df and "chi_1" in df and "chi_2" in df:
        df["chi_eff"] = (df["chi_1"] * df["mass_1"] + df["chi_2"] * df["mass_2"]) / df["total_mass"]

    # Compute luminosity distance from redshift or vice versa
    luminosity_distances = np.linspace(1, 20000, 1000)
    redshifts = np.array([cosmology.z_at_value(cosmology.luminosity_distance, dl * units.Mpc) for dl in luminosity_distances])
    z_to_dl = interp1d(redshifts, luminosity_distances, fill_value="extrapolate")
    dl_to_z = interp1d(luminosity_distances, redshifts, fill_value="extrapolate")

    df["luminosity_distance"] = df.get("luminosity_distance", z_to_dl(df["redshift"])) if "redshift" in df else df["luminosity_distance"]
    df["redshift"] = df.get("redshift", dl_to_z(df["luminosity_distance"])) if "luminosity_distance" in df else df["redshift"]

    # Convert log10 eccentricity to eccentricity
    if "log10_eccentricity" in df:
        df["eccentricity"] = 10 ** df["log10_eccentricity"]

    return df