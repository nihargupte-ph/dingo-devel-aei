import os
import numpy as np
import requests
from typing import Dict, List
from functools import partial
import glob
from tqdm import tqdm
import random
import pandas as pd
import re
from multiprocessing import Pool
from dingo.gw.domains import build_domain
import h5py
import warnings
from os.path import join
from io import StringIO
from tqdm import trange
from gwpy.timeseries import TimeSeries
import lal

from dingo.core.dataset import recursive_hdf5_save
from dingo.gw.gwutils import get_window
from dingo.gw.download_strain_data import download_psd

# NOTE TEMP
import matplotlib.pyplot as plt

"""
Contains links for PSD segment lists with quality label BURST_CAT2 from the Gravitationa Wave Open Science Center.
Some events are split up into multiple chunks such that there are multiple URLs for one observing run
"""
URL_DIRECTORY = {
    "O1_L1": [
        "https://www.gw-openscience.org/timeline/segments/O1/L1_BURST_CAT2/1126051217/11203200/"
    ],
    "O1_H1": [
        "https://www.gw-openscience.org/timeline/segments/O1/H1_BURST_CAT2/1126051217/11203200/"
    ],
    "O2_L1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/L1_BURST_CAT2/1164556817/23176801/"
    ],
    "O2_H1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/H1_BURST_CAT2/1164556817/23176801/"
    ],
    "O2_V1": [
        "https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/V1_BURST_CAT2/1164556817/23176801/"
    ],
    "O3_L1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/L1_BURST_CAT2/1238166018/15811200/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/L1_BURST_CAT2/1256655618/12708000/",
    ],
    "O3_H1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/H1_BURST_CAT2/1238166018/15811200/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/H1_BURST_CAT2/1256655618/12708000/",
    ],
    "O3_V1": [
        "https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/V1_BURST_CAT2/1238166018/15811200/",
        "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/V1_BURST_CAT2/1256655618/12708000/",
    ],
}


def get_valid_segments(segs, T_PSD, T_gap):
    """
    Given the segments `segs` and the time constraints `T_PSD`, `T_gap`, return all segments
    that can be used to estimate a PSD

    Parameters
    ----------
    segs : Tuple[int, int, int]
        Contains the start- and end gps_times as well as their difference that have been fetched from the GWOSC website
    T_PSD : int
        number of seconds used to estimate PSD
    T_gap : int
        number of seconds between two adjacent PSDs. May be negative to indicate an overlap

    Returns
    -------
    All segments that can be used to estimate a PSD
    """
    segs = np.array(segs, dtype=int)
    segs = segs[segs[:, 2] >= T_PSD]

    valid_segs = []
    for idx in range(segs.shape[0]):
        seg = segs[idx, :]
        start_time = seg[0]
        end_time = start_time + T_PSD

        while end_time in range(seg[0], seg[1] + 1):
            valid_segs.append((start_time, end_time))
            start_time = end_time + T_gap
            end_time = start_time + T_PSD

    return valid_segs


def get_path_raw_data(data_dir, run, detector, T_PSD=1024, T_gap=0):
    """
    Return the directory where the PSD data is to be stored
    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation
    T_PSD : str
        number of seconds used to estimate PSD
    T_gap : str
        number of seconds between two adjacent PSDs

    Returns
    -------
    the path where the data is stored
    """
    return os.path.join(
        data_dir, "tmp", "raw_PSDs", run, detector, str(T_PSD) + "_" + str(T_gap)
    )


def download_and_estimate_PSDs(
    data_dir: str, run: str, detector: str, settings: dict, verbose=False
):
    """
    Download segment lists from the official GWOSC website that have the BURST_CAT_2 quality label. A .npy file
    is created for every PSD that will be in the final dataset. These are stored in data_dir/tmp and may be removed
    once the final dataset has been created.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation
    settings : dict
        Dictionary of settings that are used for the dataset generation
    verbose : bool
        If true, there will be a progress bar indicating

    -------

    """

    key = run + "_" + detector
    urls = URL_DIRECTORY[key]

    starts, stops, durations = [], [], []
    for url in urls:
        r = requests.get(url, allow_redirects=True)
        c = StringIO(r.content.decode("utf-8"))
        starts_seg, stops_seg, durations_seg = np.loadtxt(c, dtype="int", unpack=True)
        starts = np.hstack([starts, starts_seg])
        stops = np.hstack([stops, stops_seg])
        durations = np.hstack([durations, durations_seg])

    T_PSD = settings["T_PSD"]
    T_gap = settings["T_gap"]
    T = settings["T"]
    f_s = settings["f_s"]

    window_kwargs = {
        "f_s": f_s,
        "roll_off": settings["window"]["roll_off"],
        "type": settings["window"]["type"],
        "T": T,
    }
    w = get_window(window_kwargs)
    # padding = window_kwargs["roll_off"]
    # seglen = window_kwargs["f_s"]*window_kwargs["T"]
    # w = lal.CreateTukeyREAL8Window(seglen, 2.0*padding*f_s / seglen)
    # w = w.data.data

    path_raw_psds = get_path_raw_data(data_dir, run, detector, T_PSD, T_gap)
    os.makedirs(path_raw_psds, exist_ok=True)

    valid_segments = get_valid_segments(
        list(zip(starts, stops, durations)), T_PSD=T_PSD, T_gap=T_gap
    )

    num_psds_max = settings["num_psds_max"]
    if num_psds_max >= 1:
        valid_segments = random.sample(valid_segments, num_psds_max)

    print(
        f"Fetching data and computing Welch's estimate of {len(valid_segments)} valid segments:\n"
    )

    for index, (start, end) in enumerate(tqdm(valid_segments, disable=not verbose)):
        filename = join(path_raw_psds, "psd_{:05d}.npy".format(index))

        if not os.path.exists(filename):
            psd = download_psd(
                det=detector,
                time_start=start,
                time_segment=T,
                window=w,
                f_s=f_s,
                num_segments=int(T_PSD / T),
            )
            np.save(
                filename,
                {
                    "detector": detector,
                    "segment": (index, start, end),
                    "time": (start, end),
                    "psd": psd,
                    "tukey window": {
                        "f_s": f_s,
                        "roll_off": settings["window"]["roll_off"],
                        "T": T,
                    },
                },
            )


def get_valid_gwf_segments(file_times, segs):
    """ 
    Parameters
    ----------
    file_times : pd.DataFrame
        Dataframe of GWF files with columns ["file_name", "gps_start", "gps_end"]
    segs : np.ndarray
        Array of valid segments from GWSOC with columns ["gps_start", "gps_end"]
    """
    # Important to sort, since iterating over segments in order
    segs = np.sort(segs, axis=0)
    file_times = file_times.sort_values(by=["gps_start"])
    intersection_segments = pd.DataFrame(columns=["file_name", "segment_start", "segment_end"])

    iter_file_times = file_times.itertuples(index=False)
    iter_valid_segments = iter(segs)
    intersection_segments = pd.DataFrame(columns=["file_name", "segment_start", "segment_end"])

    try:
        valid_start, valid_end = next(iter_valid_segments)
        file_name, file_start, file_end = next(iter_file_times)
        while True:
            overlap = (max(valid_start, file_start), min(valid_end, file_end))
            
            # If overlap is negative it must 0, so we alternatively either go to the next file or the next valid segment
            if overlap[1] - overlap[0] < 0: 
                # Figure out which interval is lower and go next on that interval
                if file_end <= valid_start:
                    file_name, file_start, file_end = next(iter_file_times)
                elif valid_end <= file_start:
                    valid_start, valid_end = next(iter_valid_segments)
                continue 

            # Recording the file name and the interval from which we want to estimate PSDs from
            df = pd.DataFrame({"file_name": [file_name], "segment_start": overlap[0], "segment_end": overlap[1]})
            intersection_segments = pd.concat([intersection_segments, df])

            # Since the iterators are sorted depending on the the overlap we have to either go to the next segment or the next file
            if overlap[0] == file_start and overlap[1] == valid_end:
                valid_start, valid_end = next(iter_valid_segments)
            elif overlap[0] == file_start and overlap[1] == file_end:
                file_name, file_start, file_end = next(iter_file_times)
            elif overlap[0] == valid_start and overlap[1] == valid_end:
                valid_start, valid_end = next(iter_valid_segments)
            elif overlap[0] == valid_start and overlap[1] == file_end:
                file_name, file_start, file_end = next(iter_file_times)
            
    except StopIteration:
        return intersection_segments

def estimate_PSDs_from_gwfs(
    data_dir: str, settings: dict, verbose=False
):
    for gwf_folder, channel, detector in zip(
        settings["gwf_folders"],
        settings["channels"],
        settings["detectors"],
    ):
        # Getting valid Segments in variable segs
        key = settings["observing_run"] + "_" + detector
        url = URL_DIRECTORY[key][0]
        r = requests.get(url, allow_redirects=True)
        c = StringIO(r.content.decode("utf-8"))
        starts_seg, stops_seg, durations_seg = np.loadtxt(c, dtype="int", unpack=True)
        segs = np.stack([starts_seg, stops_seg, durations_seg]).T
        segs = segs[segs[:, 2] >= settings["T_PSD"]]
        segs = segs[:, :2]

        # Getting gps_start and gps_end of many files 
        files = glob.glob(gwf_folder + '/**/*.gwf', recursive=True)
        file_times = pd.DataFrame(columns=["file_name", "gps_start", "gps_end"])
        for _, file in enumerate(files):
            gps_start_time, duration = re.findall(r'\b\d+\b', os.path.basename(file))
            gps_start_time, duration = float(gps_start_time), float(duration)
            df = pd.DataFrame({"file_name": [file], "gps_start": [gps_start_time], "gps_end": [gps_start_time + duration]})
            file_times = pd.concat([file_times, df])

        if file_times.empty:
            raise Exception(f"No GWF files found in {gwf_folder}")

        # Getting intersection between valid segments from GWSOC and GWF files
        intersection_segments = get_valid_gwf_segments(file_times, segs)
        # NOTE doesn't actually give you max psds rather gives you max segments
        if settings["num_segments_max"] != 0:
            intersection_segments = intersection_segments.sample(n=settings["num_segments_max"])
        
        # Iterate through intersections segments 
        for file, segment_start, segment_stop in intersection_segments.itertuples(index=False):
            seg = (segment_start, segment_stop)
            print(file, seg)
            estimate_PSDs_from_gwf(
                data_dir=data_dir, 
                gwf_filename=file, 
                channel=channel, 
                detector=detector, 
                settings=settings, 
                valid_segment=seg)




def estimate_PSDs_from_gwf(
    data_dir: str,
    gwf_filename: list,
    channel: str,
    settings: dict,
    detector: str,
    valid_segment: np.ndarray,
    verbose=False,
):
    """
    On LIGO clusters, sometimes data is stored in .gwf files and doesn't need to be downloaded from external servers. These gwf files contain strain data
    for select times in an observing run. Furthermore, LAL has it's own way of estimating the PSDs. This is what is implemented in the following function.
    In the LAL code it is something like line 946-968 of https://lscsoft.docs.ligo.org/lalsuite/lalinference/_l_a_l_inference_read_data_8c_source.html

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    channel : str
        channel that is used for the PSD dataset generation
    gwf_filename: list
        .gwf file to estimate PSDs from
    settings : dict
        Dictionary of settings that are used for the dataset generation
    detector : str
        Detector of GWF file
    valid_segment : (tuple, tuple)
        Tuple of (gps_start, gps_end) of the valid segment contained in the file. This valid segment has been cross referenced by GWSOC BURST_CAT.
    verbose : bool
        If true, there will be a progress bar indicating
    """
    T_PSD = settings["T_PSD"]
    T_gap = settings["T_gap"]
    if T_gap != 0:
        raise NotImplementedError("Non-zero T_gap not implemented")
    T = settings["T"]
    f_s = settings["f_s"]
    padding = settings["window"]["roll_off"]

    n_segs = T_PSD / T
    seglen = T * f_s

    strain = TimeSeries.read(gwf_filename, channel)
    strain = strain.resample(rate=f_s)

    long_lal_strain = lal.CreateREAL8TimeSeries(
        strain.name,
        strain.t0.value,
        0,
        strain.dt.value,
        None,
        strain.shape[0],
    )
    long_lal_strain.data.data = np.array(strain)
    times = np.arange(0,long_lal_strain.data.data.shape[0] * long_lal_strain.deltaT, long_lal_strain.deltaT) + long_lal_strain.epoch

    # We need to trim the GWF file to be the same times as the valid segment but to do this we need to know the indexes of the valid segment
    arr1 = np.abs(times - valid_segment[0])
    i = int(np.argmin(arr1))
    assert arr1[i] < 1e-3
    arr2 = np.abs(times - valid_segment[1])
    j = int(np.argmin(arr2))
    assert arr2[j] < 1e-3
    long_lal_strain = lal.ShrinkREAL8TimeSeries(long_lal_strain, i, j)

    # Dividing valid_segments into sub_segments from which we can estimate PSD
    t_length = long_lal_strain.data.data.shape[0] * long_lal_strain.deltaT
    n_psds, _ = divmod(t_length, T_PSD)
    starts = [n * int(seglen * n_segs) for n in range(int(n_psds))]
    stops = [(n + 1) * int(seglen * n_segs) for n in range(int(n_psds))]
    sub_segments = zip(starts, stops)

    path_raw_psds = get_path_raw_data(data_dir, settings["observing_run"], detector, T_PSD, T_gap)
    os.makedirs(path_raw_psds, exist_ok=True)
    last_saved_idx = max([int(s[s.index("_") + 1 : s.index(".npy")])for s in os.listdir(path_raw_psds)], default=0)

    for index, (start, end) in enumerate(tqdm(sub_segments, disable=not verbose)):
        filename = join(path_raw_psds, "psd_{:05d}.npy".format(index + last_saved_idx))

        # Copying since lal functions sometimes modify object in question
        long_lal_strain_copy = lal.CreateREAL8TimeSeries(
            long_lal_strain.name,
            long_lal_strain.epoch,
            0,
            long_lal_strain.deltaT,
            None,
            long_lal_strain.data.data.shape[0],
        )
        long_lal_strain_copy.data.data = long_lal_strain.data.data

        lal_strain = lal.ShrinkREAL8TimeSeries(long_lal_strain_copy, start, end)
        one_sided_PSD = lal.CreateREAL8FrequencySeries(
            "spectrum",
            lal_strain.epoch,
            0.0,
            f_s / seglen,
            None,
            int(seglen / 2 + 1),
        )

        # Setting up FFT structures
        time_to_freq_FFT_plan = lal.CreateForwardREAL8FFTPlan(seglen, 1)
        # Setting up window
        if 2.0 * padding * f_s / seglen < 0.0 or 2.0 * padding * f_s / seglen > 1:
            raise Exception(
                "Padding is negative or 2*padding is bigger than the whole segment. Consider reducing it or increasing segment length"
            )
        window = lal.CreateTukeyREAL8Window(seglen, 2.0 * padding * f_s / seglen)
        if settings["method"] == "Welch":
            lal.REAL8AverageSpectrumWelch(
                one_sided_PSD,
                lal_strain,
                seglen,
                int(seglen),
                window,
                time_to_freq_FFT_plan,
            )
        elif settings["method"] == "Median":
            lal.REAL8AverageSpectrumMedian(
                one_sided_PSD,
                lal_strain,
                seglen,
                int(seglen),
                window,
                time_to_freq_FFT_plan,
            )
            # Sometimes median method returns 0 PSDs
            if np.max(one_sided_PSD.data.data) < 1e-200:
                continue
        else:
            raise Exception(
                "Only options for computing PSDs are Welch and Median, please specify one of these as `method:Welch` in your settings.yaml"
            )

        # print(np.max(lal_strain.data.data))
        # if index < 3:
        #     times = np.arange(0, lal_strain.deltaT*lal_strain.data.data.shape[0], lal_strain.deltaT) + lal_strain.epoch
        #     plt.plot(times, lal_strain.data.data, label=index)
        # # freqs = np.arange(0, one_sided_PSD.deltaF*one_sided_PSD.data.data.shape[0], one_sided_PSD.deltaF) + one_sided_PSD.f0
        # # plt.plot(freqs, one_sided_PSD.data.data, label=index)
        # # print(index, np.min(one_sided_PSD.data.data))

        np.save(
            filename,
            {
                "detector": detector,
                "segment": (index, start / f_s, end / f_s),
                "time": (start / f_s, end / f_s),
                "psd": one_sided_PSD.data.data,
                "tukey window": {
                    "f_s": f_s,
                    "roll_off": padding,
                    "T": T,
                },
            },
        )



def create_dataset_from_files(
    data_dir: str, run: str, detectors: List[str], settings: dict
):

    """
    Creates a .hdf5 ASD datset file for an observing run using the estimated detector PSDs.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the ASD dataset generation
    detectors : List[str]
        Detector data that is used for the ASD dataset generation
    settings : dict
        Dictionary of settings that are used for the dataset generation
    -------
    """

    f_min = 0
    f_max = settings["f_s"] / 2
    T_PSD = settings["T_PSD"]
    T_gap = settings["T_gap"]
    T = settings["T"]

    domain_settings = {}

    save_dict = {}
    asds_dict = {}
    gps_times_dict = {}

    for ifo in detectors:

        path_raw_psds = get_path_raw_data(data_dir, run, ifo, T_PSD, T_gap)
        filenames = [el for el in os.listdir(path_raw_psds) if el.endswith(".npy")]
        psd = np.load(join(path_raw_psds, filenames[0]), allow_pickle=True).item()

        delta_f = 1 / T
        domain = build_domain(
            {
                "type": "FrequencyDomain",
                "f_min": f_min,
                "f_max": f_max,
                "delta_f": delta_f,
                "window_factor": 1.0,
            }
        )
        domain_settings["domain_dict"] = domain.domain_dict
        ind_min, ind_max = domain.min_idx, domain.max_idx

        Nf = ind_max - ind_min + 1
        asds = np.zeros((len(filenames), Nf))
        times = np.zeros(len(filenames))

        for ind, filename in enumerate(filenames):
            psd = np.load(join(path_raw_psds, filename), allow_pickle=True).item()
            asds[ind, :] = np.sqrt(psd["psd"][ind_min : ind_max + 1])
            times[ind] = psd["time"][0]

        asds_dict[ifo] = asds
        gps_times_dict[ifo] = times

    save_dict["asds"] = asds_dict
    save_dict["gps_times"] = gps_times_dict

    f = h5py.File(join(data_dir, f"asds_{run}.hdf5"), "w")
    recursive_hdf5_save(f, save_dict)
    f.attrs["settings"] = str(domain_settings)
    f.close()
