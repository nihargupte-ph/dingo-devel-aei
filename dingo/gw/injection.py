from functools import partial
import os
import jax.scipy.optimize
import numpy as np
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose
import pandas as pd
from bilby.gw.prior import BBHPriorDict
import subprocess
import inspect
import jax
import jax.numpy as jnp

from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.domains import (
    FrequencyDomain,
    build_domain,
    build_domain_from_model_metadata,
)
from dingo.gw.gwutils import (
    get_extrinsic_prior_dict,
    get_intrinsic_prior_dict,
    fill_missing_available_parameters,
)
from dingo.gw.prior import build_prior_with_defaults, split_off_extrinsic_parameters
from dingo.gw.transforms import (
    GetDetectorTimes,
    ProjectOntoDetectors,
    WhitenAndScaleStrain,
    ApplyCalibrationUncertainty,
)
from dingo.gw.waveform_generator.waveform_generator import (
    WaveformGenerator,
    NewInterfaceWaveformGenerator,
)
from dingo.core.models import PosteriorModel


class GWSignal(object):
    """
    Base class for generating gravitational wave signals in interferometers. Generates
    waveform polarizations based on provided parameters, and then projects to detectors.

    Includes option for whitening the signal based on a provided ASD.
    """

    def __init__(
        self,
        wfg_kwargs: dict,
        wfg_domain: FrequencyDomain,
        data_domain: FrequencyDomain,
        ifo_list: list,
        t_ref: float,
    ):
        """
        Parameters
        ----------
        wfg_kwargs : dict
            Waveform generator parameters [approximant, f_ref, and (optionally) f_start].
        wfg_domain : FrequencyDomain
            Domain used for waveform generation. This can potentially deviate from the
            final domain, having a wider frequency range needed for waveform generation.
        data_domain : FrequencyDomain
            Domain object for final signal.
        ifo_list : list
            Names of interferometers for projection.
        t_ref : float
            Reference time that specifies ifo locations.
        """

        self._check_domains(wfg_domain, data_domain)
        self.data_domain = data_domain

        # The waveform generator potentially has a larger frequency range than the
        # domain of the trained network / requested injection / etc. This is typically
        # the case for EOB waveforms, which require the larger range to generate
        # robustly. For this reason we have two domains.

        new_interface_flag = wfg_kwargs.get("new_interface", False)
        if new_interface_flag:
            self.waveform_generator = NewInterfaceWaveformGenerator(
                domain=wfg_domain, **wfg_kwargs
            )
        else:
            self.waveform_generator = WaveformGenerator(domain=wfg_domain, **wfg_kwargs)

        self.t_ref = t_ref
        self.ifo_list = InterferometerList(ifo_list)
        self.trigger_offset = None

        # When we set self.whiten, the projection transforms are automatically prepared.
        self._calibration_envelope = None
        self._calibration_marginalization_kwargs = None
        self.whiten = False

        self.asd = None

    @staticmethod
    def _check_domains(domain_in, domain_out):
        if domain_in.f_min > domain_out.f_min or domain_in.f_max < domain_out.f_max:
            raise ValueError(
                "Output domain is not contained within WaveformGenerator domain."
            )
        if domain_in.delta_f != domain_out.delta_f:
            raise ValueError("Domains must have same delta_f.")

    @property
    def whiten(self):
        """
        Bool specifying whether to whiten (and scale) generated signals.
        """
        return self._whiten

    @whiten.setter
    def whiten(self, value):
        self._whiten = value
        self._initialize_transform()

    @property
    def calibration_marginalization_kwargs(self):
        """
        Dictionary with the following keys:

        calibration_envelope
            Dictionary of the form {"H1": filepath, "L1": filepath, ...} with locations of
            lookup tables for the calibration uncertainty curves.

        num_calibration_nodes
            Number of nodes for the calibration model.

        num_calibration_curves
            Number of calibration curves to use in marginalization.
        """
        return self._calibration_marginalization_kwargs

    @calibration_marginalization_kwargs.setter
    def calibration_marginalization_kwargs(self, value):
        self._calibration_marginalization_kwargs = value
        self._initialize_transform()

    def _initialize_transform(self):
        transforms = [
            GetDetectorTimes(self.ifo_list, self.t_ref, self.trigger_offset),
            ProjectOntoDetectors(self.ifo_list, self.data_domain, self.t_ref),
        ]
        if self.calibration_marginalization_kwargs:
            transforms.append(
                ApplyCalibrationUncertainty(
                    self.ifo_list,
                    self.data_domain,
                    **self.calibration_marginalization_kwargs,
                )
            )
        if self.whiten:
            transforms.append(WhitenAndScaleStrain(self.data_domain.noise_std))
        self.projection_transforms = Compose(transforms)

    def signal(self, theta):
        """
        Compute the GW signal for parameters theta.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors; optionally (depending on
        self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        dict
            keys:
                waveform: GW strain signal for each detector.
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

        # Step 1: generate polarizations h_plus and h_cross
        polarizations = self.waveform_generator.generate_hplus_hcross(theta_intrinsic)
        polarizations = {  # truncation, in case wfg has a larger frequency range
            k: self.data_domain.update_data(v) for k, v in polarizations.items()
        }

        # Step 2: project h_plus and h_cross onto detectors
        sample = {
            "parameters": theta_intrinsic,
            "extrinsic_parameters": theta_extrinsic,
            "waveform": polarizations,
        }

        asd = self.asd
        if asd is not None:
            sample["asds"] = asd

        return self.projection_transforms(sample)

    # It would be good to have an ASD class to handle all of this functionality,
    # namely storing ASDs from numpy arrays, from ASDDatasets, loading from files,
    # etc. For now this functionality is partially implemented here.

    def signal_m(self, theta):
        """
        Compute the GW signal for parameters theta. Same as self.signal(theta) method,
        but it does not sum the contributions of the individual modes, and instead
        returns a dict {m: pol_m for m in [-l_max,...,0,...,l_max]} where each
        contribution pol_m transforms as ejnp(-1j * m * phase_shift) under phase shifts.

        Step 1: Generate polarizations
        Step 2: Project polarizations onto detectors;
                optionally (depending on self.whiten) whiten and scale.

        Parameters
        ----------
        theta: dict
            Signal parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        dict
            keys:
                waveform:
                    GW strain signal for each detector, with individual contributions
                    {m: pol_m for m in [-l_max,...,0,...,l_max]}
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

        # Step 1: generate m-contributions to polarizations h_plus and h_cross
        pol_m = self.waveform_generator.generate_hplus_hcross_m(theta_intrinsic)
        # truncation, in case wfg has a larger frequency range
        pol_m = {
            k_m: {
                k_pol: self.data_domain.update_data(v_pol)
                for k_pol, v_pol in v_m.items()
            }
            for k_m, v_m in pol_m.items()
        }

        # Step 2: project m-contributions to h_plus and h_cross onto detectors
        sample_out = {}
        for m, pol in pol_m.items():
            sample = {
                "parameters": theta_intrinsic,
                "extrinsic_parameters": theta_extrinsic,
                "waveform": pol,
            }
            if self.asd is not None:
                sample["asds"] = self.asd
            sample_out[m] = self.projection_transforms(sample)

        return sample_out

    @property
    def asd(self):
        """
        Amplitude spectral density.

        Either a single array, a dict (for individual interferometers),
        or an ASDDataset, from which random ASDs are drawn.
        """
        if isinstance(self._asd, np.ndarray):
            asd = {ifo.name: self._asd for ifo in self.ifo_list}
        elif isinstance(self._asd, dict):
            asd = self._asd
        elif isinstance(self._asd, ASDDataset):
            asd = self._asd.sample_random_asds()
        elif self._asd is None:
            return None
        else:
            raise TypeError("Invalid ASD type.")
        asd = {
            k: self.data_domain.update_data(v, low_value=1e-20) for k, v in asd.items()
        }
        return asd

    @asd.setter
    def asd(self, asd):
        ifo_names = [ifo.name for ifo in self.ifo_list]
        if isinstance(asd, ASDDataset):
            if set(asd.asds.keys()) != set(ifo_names):
                raise KeyError("ASDDataset ifos do not match signal.")
        elif isinstance(asd, dict):
            if set(asd.keys()) != set(ifo_names):
                raise KeyError("ASD ifos do not match signal.")
        elif isinstance(asd, str):
            raise NotImplementedError(
                "Still need to implement injections with ASDs defined by file names."
            )
        self._asd = asd


class Injection(GWSignal):
    """
    Produces injections of signals (with random or specified parameters) into stationary
    Gaussian noise. Output is not whitened.
    """

    def __init__(self, prior, **gwsignal_kwargs):
        """
        Parameters
        ----------
        prior : PriorDict
            Prior used for sampling random parameters.
        gwsignal_kwargs
            Arguments to be passed to GWSignal base class.
        """
        super().__init__(**gwsignal_kwargs)
        self.prior = prior

    @classmethod
    def from_posterior_model_metadata(cls, metadata):
        """
        Instantiate an Injection based on a posterior model. The prior, waveform
        settings, etc., will all be consistent with what the model was trained with.

        Parameters
        ----------
        metadata : dict
            Dict which you can get via PosteriorModel.metadata
        """
        intrinsic_prior = metadata["dataset_settings"]["intrinsic_prior"]
        extrinsic_prior = get_extrinsic_prior_dict(
            metadata["train_settings"]["data"]["extrinsic_prior"]
        )
        prior = build_prior_with_defaults({**intrinsic_prior, **extrinsic_prior})

        return cls(
            prior=prior,
            wfg_kwargs=metadata["dataset_settings"]["waveform_generator"],
            wfg_domain=build_domain(metadata["dataset_settings"]["domain"]),
            data_domain=build_domain_from_model_metadata(metadata),
            ifo_list=metadata["train_settings"]["data"]["detectors"],
            t_ref=metadata["train_settings"]["data"]["ref_time"],
        )

    def injection(self, theta, seed=None):
        """
        Generate an injection based on specified parameters.

        This is a signal + noise  consistent with the amplitude spectral density in
        self.asd. If self.asd is an ASDDataset, then it uses a random ASD from this
        dataset.

        Data are not whitened.

        Parameters
        ----------
        theta : dict
            Parameters used for injection.

        Returns
        -------
        dict
            keys:
                waveform: data (signal + noise) in each detector
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        signal = self.signal(theta)
        try:
            # Be careful to use the ASD included with the signal, since each time
            # self.asd is accessed it gives a different ASD (if using an ASD dataset).
            asd = signal["asds"]
        except KeyError:
            raise ValueError("self.asd must be set in order to produce injections.")

        if self.whiten:
            print("self.whiten was set to True. Resetting to False.")
            self.whiten = False

        if seed is not None:
            np.random.seed(seed)
        data = {}
        for ifo, s in signal["waveform"].items():
            noise = (
                (np.random.randn(len(s)) + 1j * np.random.randn(len(s)))
                * asd[ifo]
                * self.data_domain.noise_std
            )
            d = s + noise
            data[ifo] = self.data_domain.update_data(d, low_value=0.0)

        signal["waveform"] = data
        return signal

    def random_injection(self):
        """
        Generate a random injection.

        This is a signal + noise  consistent with the amplitude spectral density in
        self.asd. If self.asd is an ASDDataset, then it uses a random ASD from this
        dataset.

        Data are not whitened.

        Returns
        -------
        dict
            keys:
                waveform: data (signal + noise) in each detector
                extrinsic_parameters: {}
                parameters: waveform parameters
                asd (if set): amplitude spectral density for each detector
        """
        theta = self.prior.sample()
        theta = {
            k: float(v) for k, v in theta.items()
        }  # Some parameters are np.float64
        return self.injection(theta)


class HyperInjection(object):
    """create a set of asimov injections to analyze"""

    def __init__(
        self,
        model,
        parameter_grids,
        model_filepath_list,
        p_det = None,
        network_conditions=None
    ):
        """
        Parameters
        ----------
        model : gwpopulation.ejnperimental.jax.NonCachingModel

        parameters_grids : dict
            Dictionary of parameter names and a grid 
            over which we will sample the parameters. 

        model_filepath_list : list[dict]
            List of filepaths to the networks to be used in the analysis.
            For example,
            [
                {"model": "path/to/main_network.pt", "model_init": "path/to/init_network.pt"}, # gnpe
                {"model": "path/to/main_network.pt"}, # non-gnpe
            ]

        p_det : function (optional)
            Function which takes in a set of parameters as a dict
            or dataframe as an input and returns the the probability
            of detecting a signal with those parameters.

        network_conditions : dict[func[dict] -> bool] (optional)
            Dictionary of functions which take in a set of parameters
            as a dict and return a boolean. Each function corresponds to a 
            function that has been passed in with model_filepath_list. The output
            of the function determines whether or not to use the network. The functions
            should be mutually exclusive, that is, two outputs cannot be true at the same 
            time. 
        """
        self.model = model
        self.parameter_grids = parameter_grids
        if network_conditions is not None:
            assert len(network_conditions) == len(model_filepath_list)
            self.network_conditions = network_conditions    
            
        if p_det is None:
            self.p_det = lambda x: np.ones(len(x))
        else:
            self.p_det = p_det

        self.hyper_parameter_names = []
        for model in self.model.models:
            if inspect.isclass(model):
                if hasattr(model, "primary_model"):
                    hyper_parameter_names = inspect.getfullargspec(model.primary_model)[0][1:]

                    # NOTE TEMP hardcoded for now
                    hyper_parameter_names.insert(-1, "delta_m")
                elif hasattr(model, "variable_names"):
                    hyper_parameter_names = model.variable_names
            else:
                hyper_parameter_names = inspect.getfullargspec(model)[0][1:]
            # if starts with "dataset" key, then the model uses a dict input
            # otherwise it ejnpects an arraylike input
            self.hyper_parameter_names.append(hyper_parameter_names)

        self.model_filepath_list = model_filepath_list
        self.network_prior_list = []
        self.inference_parameters = []
        for d in model_filepath_list:
            pm = PosteriorModel(
                device="cuda",
                model_filename=d["model"],
                load_training_info=True,
            )
            intrinsic_prior = get_intrinsic_prior_dict(
                pm.metadata["dataset_settings"]["intrinsic_prior"]
            )
            # delete any default parameters which may not be inferred
            intrinsic_prior = {
                k: v
                for k, v in intrinsic_prior.items()
                if k in pm.metadata["dataset_settings"]["intrinsic_prior"].keys()
            }

            # add inferrable parameters to inference parameters
            self.inference_parameters.append(
                pm.metadata["train_settings"]["data"]["inference_parameters"]
            )

            # check if any default priors are set in the intrinsic prior
            extrinsic_prior = get_extrinsic_prior_dict(
                pm.metadata["train_settings"]["data"]["extrinsic_prior"]
            )
            prior = BBHPriorDict({**intrinsic_prior, **extrinsic_prior})

            self.network_prior_list.append(prior)

        # check all inference parameters of all networks are the same
        if not all(
            x == self.inference_parameters[0] for x in self.inference_parameters
        ):
            print(f"""Warning, inference parameters are not the same for all networks. Defaulting
                  to {self.inference_parameters[0]}""")
            self.inference_parameters = self.inference_parameters[0]
        else:
            self.inference_parameters = self.inference_parameters[0]

    def find_correct_network_idx(self, sample):
        """
        Find the correct network to use for the analysis
        based off the prior. Will take the network which has
        the highest log_prior value.

        Parameters
        ----------
        sample : dict
            Dict of parameters

        Returns
        -------
        network_idx : int
            Index of the network to use for the analysis
        """
        if hasattr(self, "network_conditions"):
            network_idx = np.where([condition(sample) for condition in self.network_conditions])[0][0]
        else:
            log_prior = [prior.ln_prob(sample) for prior in self.network_prior_list]
            network_idx = np.argmax(log_prior)
            
        return network_idx

    def sample_injection_parameters(
        self,
        hyper_injection_parameters,
        num_injections,
        sampling_algorithm="inverse_transform_sampling",
    ):
        """
        Parameters
        ----------
        hyper_injection_parameters : list
            Dictionary of hyper parameters to be used for injections.
            Should correspond to each submodel of model. That is it
            should be of the form
            [
                {"model_1_hyper_parameter_1": val_1, ...},
                {"model_2_hyper_parameter_1": val_1, ...},
                ...
            ]

        num_injections : int
            Number of injections to generate.

        sampling_algorithm : str default="inverse_transform_sampling"
            Algorithm to use for sampling from the distribution.
            Currently implemented are "inverse_transform_sampling" and "metropolis_hastings"
        

        Sample injection parameters based on model

        """
        sampling_algorithm = eval(sampling_algorithm)

        sampled_parameters = []
        # iterating through models based off the hyper parameters
        # sampli
        # ng the sub parameters
        for i, sub_model in enumerate(self.model.models):
            partial_prob = partial(sub_model, **hyper_injection_parameters[i])

            # redefine the target density to accept numeric types
            param_names = list(self.parameter_grids[i].keys())

            # if ejnpecting a dict input, then we need to redefine the target density
            def target_density(args):
                # TODO the _src replacement is a hack to get around the fact that 
                # gwpop models use src frame masses but don't call them mass_1_src for example
                return partial_prob({param_names[j].replace("_src", ""): args[j] for j in range(len(args))})

            # grid to sample over
            sub_parameters_grid = [
                self.parameter_grids[i][param_name]
                for param_name in param_names
            ]

            sampled_parameters.append(
                pd.DataFrame(
                    sampling_algorithm(
                        target_density, sub_parameters_grid, num_samples=num_injections 
                    ),
                    columns=param_names,
                )
            )

        injection_samples = pd.concat(sampled_parameters, axis=1)

        # Every population model may not parametrize the all the parameters
        # of a waveform model. For example, a population model may only parametrize
        # the chirp_mass, but we still need to choose the right ascension
        # to generate an injection. This function will randomly sample missing
        # parameters from the prior. This is marginalized over when doing the 
        # hierarchicial inference. 
        # First get any parameters which can be derived from the already sampled parameters
        injection_samples = fill_missing_available_parameters(injection_samples)

        # filling in other parameters by sampling from the prior
        prior_samples = pd.DataFrame(
            self.network_prior_list[0].sample(size=num_injections)
        )
        missing_keys = list(set(prior_samples.keys()) - set(injection_samples.keys()))
        injection_samples = pd.concat(
            [injection_samples, prior_samples[missing_keys]], axis=1
        )

        # call this again to fill in any missing parameters that come from 
        # sampling from the prior
        injection_samples = fill_missing_available_parameters(injection_samples)
        
        return injection_samples

    def apply_selection_criteria(self, injection_samples):
        """
        Include the effect of malmquist bias/selection effects

        Parameters
        ----------
        injection_samples : pd.DataFrame
            Samples of the injection parameters.
        """
        sampled_p_det = self.p_det(injection_samples)
        random_samples = np.random.uniform(0, 1, len(sampled_p_det))
        selection_mask = np.array(random_samples < sampled_p_det)

        return injection_samples[selection_mask]

    def generate_hyper_injection_inis(
        self, injection_samples, out_folder, dingo_pipe_kwargs={}
    ):
        """
        Based on the available networks, create a
        set of dingo pipe analyses. Will dynamically adjust the 
        used networks based on the provided criteria.

        Parameters
        ----------
        injection_samples : pd.DataFrame
            Samples of the injection parameters.
        out_folder : str
            Folder to save the injection .ini files and
            dingo pipe runs.
        """
        # subsetting the passed parameters to be ones compatible
        # with the network and the prior

        dingo_pipe_settings = default_dingo_pipe_config.copy()
        dingo_pipe_settings.update(dingo_pipe_kwargs)

        # making the directories to save the injection .ini files
        os.makedirs(out_folder, exist_ok=True)

        for i, injection_sample in injection_samples.iterrows():
            # subsetting the passed parameters to be ones compatible
            sub_sample = injection_sample[
                [*self.inference_parameters, "phase"]
            ].to_dict()

            dingo_pipe_settings["outdir"] = os.path.join(out_folder, str(i))
            dingo_pipe_settings["injection-dict"] = sub_sample
            dingo_pipe_settings["label"] = f"injection_{i}"

            network_dict = self.model_filepath_list[
                self.find_correct_network_idx(sub_sample)
            ]
            dingo_pipe_settings.update(network_dict)

            # save the .ini file
            with open(os.path.join(out_folder, f"injection_{i}.ini"), "w+") as f:
                for k, v in dingo_pipe_settings.items():
                    f.write(f"{k}={v}\n")

    def generate_hyper_injection(
        self,
        hyper_injection_parameters,
        num_injections,
        out_folder,
        dingo_pipe_kwargs={},
        submit=False,
    ):
        """
        Parameters
        ----------
        hyper_injection_parameters : list
            Dictionary of hyper parameters to be used for injections.
            Should correspond to each submodel of model. That is it
            should be of the form
            [
                {"model_1_hyper_parameter_1": val_1, ...},
                {"model_2_hyper_parameter_1": val_1, ...},
                ...
            ]


        num_injections : int
            Number of injections to generate.

        out_folder : str
            Folder to save the injection .ini files and
            dingo pipe runs.

        dingo_pipe_kwargs : dict default=None
            Dictionary of dingo pipe parameters to be used for the analysis.
            This will update the default_dingo_pipe_config.

        submit : bool default=False
            Whether to submit the injection dags to the cluster.

        Based on the available networks, create a
        set of dingo pipe analyses. The pipeline is

        1). Sample injection parameters
        2). Fill in missing parameters
        3). Apply selection criteria
        4). Generate injection .ini files
        6). Submit injection .ini files
        """
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        os.chdir(out_folder)

        selected_injection_samples = pd.DataFrame([])
        while len(selected_injection_samples) < num_injections:
            # generating more injection samples than we need in case 
            # some are rejected
            injection_samples = self.sample_injection_parameters(
                hyper_injection_parameters=hyper_injection_parameters,
                num_injections=num_injections * 50,
            )
            tmp_selected_injection_samples = self.apply_selection_criteria(injection_samples)
            sub_samples = tmp_selected_injection_samples.sample(num_injections - len(selected_injection_samples), replace=False)
            selected_injection_samples = pd.concat([selected_injection_samples, sub_samples])
        selected_injection_samples = selected_injection_samples.head(num_injections)

        self.generate_hyper_injection_inis(
            selected_injection_samples, out_folder, dingo_pipe_kwargs=dingo_pipe_kwargs
        )
        if submit:
            submit_hyper_injection_inis(out_folder)
        else:
            print("Injections ini's in ", out_folder)
        
        return selected_injection_samples

def metropolis_hastings(target_density, sub_parameters_min_max, num_samples):
    """
    Metropolis-Hastings algorithm for sampling over the
    hyper probability.

    Parameters
    ----------
    prob_function : function
        The probability function to be sampled over.

    min_max : dict
    """

    burnin_size = 500_000
    size = burnin_size + num_samples

    # defining the initial point as the midpoint of the min-max
    x0 = pd.DataFrame(
        {k: (b + a) / 2 for k, (a, b) in sub_parameters_min_max.items()}, index=[0]
    )
    xt = x0
    step_size = pd.DataFrame(
        {k: (b - a) / 50 for k, (a, b) in sub_parameters_min_max.items()}, index=[0]
    )

    samples = []
    num_accepted_samples = 0
    for i in range(size):
        perturbation = pd.DataFrame(
            {k: np.random.normal(loc=0, scale=step) for k, step in step_size.items()},
            index=[0],
        )
        xt_candidate = xt + perturbation

        accept_prob = (target_density(xt_candidate)) / (target_density(xt))

        if np.random.uniform(0, 1) < accept_prob.iloc[0]:
            xt = xt_candidate
            num_accepted_samples += 1

        samples.append(xt)

    print(target_density, "Acceptance rate: ", num_accepted_samples / size)
    samples = pd.concat(samples[burnin_size:], ignore_index=True)
    return samples

def inverse_transform_sampling(target_density, grids, num_samples, batch_size=10_000):

    """
    Inverse transform sampling algorithm for sampling over the
    hyper probability.

    Parameters
    ----------
    prob_function : function(float) -> float
        The probability function to be sampled over.

    grids : list[jnp.ndarray]
        List of grids to integrate the cdf over

    num_samples : int
        Number of samples to generate.
    """

    key1 = jax.random.PRNGKey(np.random.randint(0, 1e6))
    key2 = jax.random.PRNGKey(np.random.randint(0, 1e6))

    # JIT compile the target density function
    # jit_target_density = jax.jit(target_density) # doesn't work for all gwpop functions for some
    # reason
    
    if len(grids) == 1:
        grid = grids[0]
        grid = jax.device_put(grid)
        cdf = jnp.cumsum(target_density([grid])[:-1] * jnp.diff(grid))
        cdf = cdf / cdf[-1]
        uniform_samples = jax.random.uniform(key1, shape=(num_samples,))
        idx = jnp.argmin(jnp.abs(cdf - uniform_samples[:, None]), axis=1)
        return grid[idx]
    
    elif len(grids) == 2:
        # Step 1: Compute the p(x, y) on a grid
        grid1, grid2 = grids
        grid1, grid2 = jax.device_put(grid1), jax.device_put(grid2)
        X, Y = jnp.meshgrid(grid1, grid2, indexing='ij')

        # Flatten the meshgrid to prepare for batching
        X_flat, Y_flat = X.ravel(), Y.ravel()
        points = jnp.stack([X_flat, Y_flat], axis=-1)

        # Compute densities in blocks
        density_values_flat = jax.lax.map(target_density, points, batch_size=batch_size)

        # Reshape back to grid shape
        density_values = density_values_flat.reshape(X.shape)

        # Step 2: marginalize over x to compute p(y)
        # compute the marginal density for each sample
        marginal_density = jnp.sum(density_values, axis=0)

        # Handle infinities
        inf_mask = jnp.isinf(marginal_density)
        max_value = jnp.max(jnp.where(inf_mask, -jnp.inf, marginal_density))
        marginal_density = jnp.where(inf_mask, max_value, marginal_density)

        # Step 3: Compute the CDF of c(Y) = \int_0^Y p(y) dy
        # Compute marginal CDF
        marginal_cdf = jnp.cumsum(marginal_density)[:-1]
        marginal_cdf = marginal_cdf / marginal_cdf[-1]
        
        # sample points from the marginal CDF
        uniform_samples = jax.random.uniform(key1, shape=(num_samples,))
        idx2 = jnp.argmin(jnp.abs(marginal_cdf - uniform_samples[:, None]), axis=1)
        var2_samples = grid2[idx2]

        # Step 4: Compute the conditional CDF of c(X|Y) = \int_0^X p(x|y) dx
        # Conditional density
        conditional_densities = density_values / marginal_density
        conditional_density = jnp.take(conditional_densities, idx2, axis=1).T

        # Step 5: Sample from the conditional CDF
        # Conditional CDF
        conditional_cdfs = jnp.cumsum(conditional_density, axis=1)
        conditional_cdfs = conditional_cdfs / conditional_cdfs[:, -1, None]

        uniform_samples = jax.random.uniform(key2, shape=(num_samples,))
        idx1 = jnp.argmin(jnp.abs(conditional_cdfs - uniform_samples[:, None]), axis=1)
        var1_samples = grid1[idx1]

        return jnp.stack([var1_samples, var2_samples], axis=1)

def submit_hyper_injection_inis(out_folder):
    """
    Run dingo_pipe and submit the dags
    """

    # running dingo_pipe
    for ini_file in os.listdir(out_folder):
        if ini_file.endswith(".ini"):
            os.system(f"dingo_pipe {os.path.join(out_folder, ini_file)}")

    # submitting the dag
    env = {}
    env.update(os.environ)
    for root, _, files in os.walk(out_folder):
        for file in files:
            if file.startswith("dag"):
                dag_fpath = os.path.join(root, file)
                command = ["condor_submit_dag", dag_fpath]
                output = subprocess.check_output(command, env=env)

default_dingo_pipe_config = {
    "local": False,
    "accounting": "dingo",
    "request-cpus-importance-sampling": 32,
    "n-parallel": 20,
    "request-memory": 120,
    "request-memory-generation": 8.0,
    "request-disk": 0.5,
    "sampling-requirements": "[TARGET.CUDAGlobalMemoryMb>20000]",
    "extra-lines": "[getenv=True]",
    "simple-submission": False,
    "model_init": "model_init.pt",
    "model": "model.pt",
    "device": "cuda",
    "num-gnpe-iterations": 30,
    "num-samples": 50_000,
    "batch-size": 50_000,
    "recover-log-prob": True,
    "trigger-time": 1267963151.3,
    "shift-segment-for-psd-generation-if-nan": True,
    "label": "GW150914",
    "outdir": "./outdir_GW150914",
    "channel-dict": {"H1": "GWOSC", "L1": "GWOSC"},
    "psd-length": 128,
    "plot-corner": True,
    "plot-weights": True,
    "plot-log-probs": True,
    "local-generation": True,
    "environment-variables": {"NO_GETCONF":True}
}
