# Quickstart tutorial

To learn to use Dingo, we recommend starting with the examples provided in the `/examples`
folder. The YAML files contained in this directory (and subdirectories) contain
configuration settings for the various Dingo tasks (constructing training data, training networks, and performing inference). These files should be provided as input to the
command-line scripts, which then run Dingo and save output files. These output files
contain as metadata the settings in the `.yaml` files, and they may usually be inspected
by running `dingo_ls`.

```{mermaid}
flowchart TB
    dataset_settings[dataset_settings.yaml]
    dataset_settings-->generate_dataset(["dingo_generate_dataset
    #nbsp; #nbsp; --settings_file dataset_settings.yaml
    #nbsp; #nbsp; --out_file waveform_dataset.hdf5"])
    style generate_dataset text-align:left
    asd_settings[asd_dataset_settings.yaml]
    asd_settings-->generate_asd(["generate_asd_dataset
    #nbsp; #nbsp; --settings_file dataset_settings.yaml
    #nbsp; #nbsp; --data_dir asd_dataset"])
    style generate_asd text-align:left
    train_init(["dingo_train 
    #nbsp; #nbsp; --settings_file train_settings_init.yaml
    #nbsp; #nbsp; --train_dir model_init"])
    style train_init text-align:left
    train_settings_init[train_settings_init.yaml]
    train_settings_init-->train_init
    generate_dataset--->train_init
    generate_asd--->train_init
    generate_dataset--->train_main(["dingo_train 
    #nbsp; #nbsp; --settings_file train_settings_main.yaml
    #nbsp; #nbsp; --train_dir model_main"])
    style train_main text-align:left
    train_settings_main[train_settings_main.yaml]
    generate_asd--->train_main
    train_settings_main-->train_main
    train_init-->inference(["dingo_analyze_event
    #nbsp; #nbsp; --model model_main/model_stage_1.pt
    #nbsp; #nbsp; --model_init model_init/model_stage_1.pt
    #nbsp; #nbsp; --num_samples 50000
    #nbsp; #nbsp; --gps_time_event 1126259462.4"])
    style inference text-align:left
    train_main-->inference
    inference-->samples[dingo_samples-1126259462.4.hdf5]
```




After configuring the settings files, the scripts may be used as follows, assuming the
Dingo `venv` is active.

## Generate training data

### Waveforms

To generate a waveform dataset for training, execute

```
dingo_generate_dataset --settings_file waveform_dataset_settings.yaml --num_processes N --out_file waveform_dataset.hdf5
```

where `N` is the number of processes you would like to use to generate the waveforms in
parallel. This saves the dataset of waveform polarizations in the
file `waveform_dataset.hdf5` (typically compressed using SVD, depending on configuration).

One can use `dingo_generate_dataset_dag` to set up a condor DAG for generating waveforms
on a cluster. This is typically useful for slower waveform models.

### Noise ASDs

Training also requires a dataset of noise ASDs, which are sampled randomly for each
training sample. To generate this dataset based on noise observed during a run, execute

```
dingo_generate_ASD_dataset --data_dir data_dir --settings_file asd_dataset_settings.yaml
```

This will download data from the GWOSC website and create a `/tmp` directory, in which the
estimated PSDs are stored. Subsequently, these are collected together into a final `.hdf5`
ASD dataset.
If no `settings_file` is passed, the script will attempt to use the default
one `data_dir/asd_dataset_settings.yaml`.

## Training

With a waveform dataset and ASD dataset(s), one can train a neural network. Configure
the `train_settings.yaml` file to point to these datasets, and run

```
dingo_train --settings_file train_settings.yaml --train_dir train_dir
```

This will configure the network, train it, and store checkpoints, a record of the history,
and the final network in the directory `train_dir`. Alternatively, to resume training from
a checkpoint file, run

```
dingo_train --checkpoint model.pt --train_dir train_dir
```

If using CUDA on a machine with several GPUs, be sure to first select the desired GPU
number using the `CUDA_VISIBLE_DEVICES` environment variable. If using a cluster, Dingo
can be trained using `dingo_train_condor`.

## Inference

Once a Dingo model is trained, inference for real events can be performed with

```
dingo_analyze_event
  --model model.pt
  --gps_time_event gps_time_event
  --num_samples num_samples
  --batch_size batch_size
```

where `model.pt` is the path of the trained Dingo mode, `gps_time_event` is the GPS time
of the event to be analyzed (e.g., 1126259462.4 for GW150914), `num_samples` is the number
of desired samples and `batch_size` is the batch size (the larger the faster the
computation, but limited by GPU memory). Dingo downloads the event data from GWOSC. It
also estimates the noise ASD from data prior to the event.

If Dingo was trained using GNPE (with the `data/gnpe_time_shifts` option in the settings
file) then one must train an additional Dingo model to initialize the Gibbs sampler. This
model infers initial estimates for the coalescence times in the individual detectors and
is trained just like any other dingo model. See `training/train_settings_init.yaml` for an
example settings file. To perform inference using GNPE, the script must be pointed to this
model:

```
dingo_analyze_event
  --model model
  --model_init model_init
  --gps_time_event gps_time_event
  --num_samples num_samples
  --num_gnpe_iterations num_gnpe_iterations
  --batch_size batch_size
```

where `model_init` is the path of the aforementioned initialization model,
and `num_gnpe_iterations` specifies the number of GNPE iterations (
typically, `num_gnpe_iterations=30`).

Finally, the option `--event_dataset </path/to/event_dataset.hdf5>` can be set to cache
downloaded event data for future use.

## Importance sampling

To perform importance sampling, run

`python dingo/gw/importance_sampling/importance_weights.py --settings is_settings.yaml`

where `is_settings.yaml` contains the settings, and in particular points to the output
file that was previously generated by Dingo.

[//]: # (The quickest way to get started with Dingo is to follow the examples in the repository.)

[//]: # ()

[//]: # (Running Your First Injection)

[//]: # (----------------------------)

[//]: # ()

[//]: # (A general pipeline to using dingo for inference on injections is to )

[//]: # ()

[//]: # (1. Generate a :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset` )

[//]: # (2. Generate a :class:`~dingo.gw.ASD_dataset.noise_dataset.ASDDataset`)

[//]: # (3. Generate and train a :class:`~dingo.core.models.posterior_model.PosteriorModel` using the :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset`  and :class:`~dingo.gw.ASD_dataset.noise_dataset.ASDDataset` )

[//]: # (4. Generate a :class:`~dingo.gw.inference.gw_samplers.GWSampler` using the trained :class:`~dingo.core.models.posterior_model.PosteriorModel` to do inference on a :class:`~dingo.gw.inference.injection.Injection`)

[//]: # ()

[//]: # ()

[//]: # (This tutorial will take you through how to start with various settings files and go through steps 1-4. At the end you will be able to generate a corner plot of an injection using dingo!)

[//]: # ()

[//]: # ()

[//]: # (Step 1, Generating a :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset` )

[//]: # (------------------------------------)

[//]: # ()

[//]: # (Generating a :class:`~dingo.gw.dataset.waveform_dataset.WaveformDataset` is largely done with the use of a a `settings.yaml` file. You can edit this file to change the )

[//]: # (priors, waveform approximant, f_max etc. Here is a sample settings.yaml file. )

[//]: # ()

[//]: # ()

[//]: # (.. literalinclude:: ../../tutorials/02_gwpe/datasets/waveforms/settings.yaml)

[//]: # (   :language: yaml)

[//]: # ()

[//]: # ()

[//]: # ()

[//]: # (Dingo's functionality is largely wrapped around the :class:`~dingo.core.models.posterior_model.PosteriorModel` class. This is the class which )