import copy
import math
import time
from pathlib import Path
from typing import Optional, Union
import sys

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose

from dingo.core.models import PosteriorModel
from dingo.core.result import Result
from dingo.core.result import DATA_KEYS as RESULT_DATA_KEYS
from dingo.core.utils import torch_detach_to_cpu, IterationTracker

# FIXME: transform below should be in core
from dingo.gw.transforms import SelectStandardizeRepackageParameters

#
# Sampler classes are motivated by the approach of Bilby.
#


class Sampler(object):
    """
    Sampler class that wraps a PosteriorModel. Allows for conditional and unconditional
    models.

    Draws samples from the model based on (optional) context data.

    This is intended for use either as a standalone sampler, or as a sampler producing
    initial sample points for a GNPE sampler.

    Methods
    -------
    run_sampler
    log_prob
    to_result
    to_hdf5

    Attributes
    ----------
    model : PosteriorModel
    inference_parameters : list
    samples : DataFrame
        Samples produced from the model by run_sampler().
    context : dict
    metadata : dict
    event_metadata : dict
    unconditional_model : bool
        Whether the model is unconditional, in which case it is not provided context
        information.
    transform_pre, transform_post : Transform
        Transforms to be applied to data and parameters during inference. These are
        typically implemented in a subclass.
    """

    def __init__(
        self,
        model: PosteriorModel,
    ):
        """
        Parameters
        ----------
        model : PosteriorModel
        """
        self.model = model

        self.metadata = self.model.metadata.copy()
        if self.metadata["train_settings"]["data"].get("unconditional", False):
            self.unconditional_model = True
            # For unconditional models, the context will be stored with the model. It
            # is needed for calculating the likelihood for importance sampling.
            # However, it will not be used when sampling from the model, since it is
            # unconditional.
            self.context = self.model.context
            self.event_metadata = self.model.event_metadata
            self.base_model_metadata = self.metadata["base"]
        else:
            self.unconditional_model = False
            self.context = None
            self.event_metadata = None
            self.base_model_metadata = self.metadata

        self.inference_parameters = self.metadata["train_settings"]["data"][
            "inference_parameters"
        ]

        self.samples = None
        self._build_domain()
        # Must be after _build_domain() since transforms can depend on domain.
        self._initialize_transforms()

        self._pesummary_package = "core"
        self._result_class = Result

    @property
    def context(self):
        """Data on which to condition the sampler. For injections, there should be a
        'parameters' key with truth values."""
        return self._context

    @context.setter
    def context(self, value):
        if value is not None and "parameters" in value:
            self.metadata["injection_parameters"] = value.pop("parameters")
        self._context = value

    @property
    def event_metadata(self):
        """Metadata for data analyzed. Can in principle influence any post-sampling
        parameter transformations (e.g., sky position correction), as well as the
        likelihood detector positions."""
        return self._event_metadata

    @event_metadata.setter
    def event_metadata(self, value):
        self._event_metadata = value

    def _initialize_transforms(self):
        self.transform_pre = Compose([])

        # De-standardize data and extract inference parameters. This needs to be here
        # (and not just in subclasses) in order for run_sampler() to properly execute.
        self.transform_post = SelectStandardizeRepackageParameters(
            {"inference_parameters": self.inference_parameters},
            self.metadata["train_settings"]["data"]["standardization"],
            inverse=True,
            as_type="dict",
        )

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
    ) -> dict:
        if not self.unconditional_model:
            if context is None:
                raise ValueError("Context required to run sampler.")
            x = context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # transforms_pre are expected to transform the data in the same way for each
            # requested sample. We therefore expand it across the batch *after*
            # pre-processing.
            x = self.transform_pre(context)
            x = x.expand(num_samples, *x.shape)
            x = [x]
            # The number of samples is expressed via the first dimension of x,
            # so we must pass num_samples = 1 to sample_and_log_prob().
            num_samples = 1
        else:
            if context is not None:
                print("Unconditional model. Ignoring context.")
            x = []

        # For a normalizing flow, we get the log_prob for "free" when sampling,
        # so we always include this. For other architectures, it may make sense to
        # have a flag for whether to calculate the log_prob.
        self.model.model.eval()
        with torch.no_grad():
            y, log_prob = self.model.model.sample_and_log_prob(
                *x, num_samples=num_samples
            )

        samples = self.transform_post({"parameters": y, "log_prob": log_prob})
        result = samples["parameters"]
        result["log_prob"] = samples["log_prob"]
        return result

    def run_sampler(
        self,
        num_samples: int,
        batch_size: Optional[int] = None,
    ):
        """
        Generates samples and stores them in self.samples. Conditions the model on
        self.context if appropriate (i.e., if the model is not unconditional).

        If possible, it also calculates the log_prob and saves it as a column in
        self.samples. When using GNPE it is not possible to obtain the log_prob due to
        the many Gibbs iterations. However, in the case of just one iteration, and when
        starting from a sampler for the proxy, the GNPESampler does calculate the
        log_prob.

        Allows for batched sampling, e.g., if limited by GPU memory. Actual sampling
        for each batch is performed by _run_sampler(), which will differ for Sampler
        and GNPESampler.

        Parameters
        ----------
        num_samples : int
            Number of samples requested.
        batch_size : int, optional
            Batch size for sampler.
        """
        self.samples = None

        print(f"Running sampler to generate {num_samples} samples.")
        t0 = time.time()
        if not self.unconditional_model:
            if self.context is None:
                raise ValueError("Context must be set in order to run sampler.")
            context = self.context
        else:
            context = None

        # Carry out batched sampling by calling _run_sample() on each batch and
        # consolidating the results.
        if batch_size is None:
            batch_size = num_samples
        full_batches, remainder = divmod(num_samples, batch_size)
        samples = [self._run_sampler(batch_size, context) for _ in range(full_batches)]
        if remainder > 0:
            samples.append(self._run_sampler(remainder, context))
        samples = {p: torch.cat([s[p] for s in samples]) for p in samples[0].keys()}
        samples = {k: v.cpu().numpy() for k, v in samples.items()}

        # Apply any post-sampling transformation to sampled parameters (e.g.,
        # correction for t_ref) and represent as DataFrame.
        self._post_process(samples)
        self.samples = pd.DataFrame(samples)
        print(f"Done. This took {time.time() - t0:.1f} s.")
        sys.stdout.flush()

    def log_prob(self, samples: pd.DataFrame) -> np.ndarray:
        """
        Calculate the model log probability at specific sample points.

        Parameters
        ----------
        samples : pd.DataFrame
            Sample points at which to calculate the log probability.

        Returns
        -------
        np.array of log probabilities.
        """
        # TODO: Check / fix this method. It is likely broken, but is not critical.
        if self.context is None and not self.unconditional_model:
            raise ValueError("Context must be set in order to calculate log_prob.")

        # This undoes any post-correction that would have been done to the samples,
        # before evaluating the log_prob. E.g., the t_ref / sky position correction.
        samples = samples.copy()
        self._post_process(samples, inverse=True)

        # Standardize the sample parameters and place on device.
        y = samples[self.inference_parameters].to_numpy()
        standardization = self.metadata["train_settings"]["data"]["standardization"]
        mean = np.array([standardization["mean"][p] for p in self.inference_parameters])
        std = np.array([standardization["std"][p] for p in self.inference_parameters])
        y = (y - mean) / std
        y = torch.from_numpy(y).to(device=self.model.device, dtype=torch.float32)

        if not self.unconditional_model:
            x = self.context.copy()
            x["parameters"] = {}
            x["extrinsic_parameters"] = {}

            # Context is the same for each sample. Expand across batch dimension after
            # pre-processing.
            x = self.transform_pre(self.context)
            x = x.expand(len(samples), *x.shape)
            x = [x]
        else:
            x = []

        self.model.model.eval()
        with torch.no_grad():
            log_prob = self.model.model.log_prob(y, *x)

        log_prob = log_prob.cpu().numpy()
        log_prob -= np.sum(np.log(std))

        # Pre-processing step may have included a log_prob with the samples.
        if "log_prob" in samples:
            log_prob += samples["log_prob"].to_numpy()

        return log_prob

    def _post_process(self, samples: Union[dict, pd.DataFrame], inverse: bool = False):
        pass

    def _build_domain(self):
        self.domain = None

    def _build_likelihood(self, **likelihood_kwargs):
        self.likelihood = None

    def importance_sample(self, num_processes: int = 1, **likelihood_kwargs):
        """
        Calculate importance weights for samples.

        Importance sampling starts with samples have been generated from a proposal
        distribution q(theta), in this case a neural network model. Certain networks
        (i.e., non-GNPE) also provide the log probability of each sample,
        which is required for importance sampling.

        Given the proposal, we re-weight samples according to the (un-normalized)
        target distribution, which we take to be the likelihood L(theta) times the
        prior pi(theta). This gives sample weights

            w(theta) ~ pi(theta) L(theta) / q(theta),

        where the overall normalization does not matter (and we take to have mean 1).
        Since q(theta) enters this expression, importance sampling is only possible
        when we know the log probability of each sample.

        As byproducts, this method also estimates the evidence and effective sample
        size of the importance sampled points.

        This method modifies the samples pd.DataFrame in-place, adding new columns for
        log_likelihood, log_prior, and weights. It also stores log_evidence,
        effective_sample_size and n_eff attributes.

        Parameters
        ----------
        num_processes : int
            Number of parallel processes to use when calculating likelihoods. (This is
            the most expensive task.)
        likelihood_kwargs : dict
            kwargs that are forwarded to the likelihood constructor. E.g., options for
            marginalization.
        """

        if self.samples is None:
            raise KeyError(
                "Initial samples are required for importance sampling. "
                "Please execute run_sampler()."
            )
        if "log_prob" not in self.samples:
            raise KeyError(
                "Stored samples do not contain log probability, which is "
                "needed for importance sampling."
            )

        self._build_likelihood(**likelihood_kwargs)

        # Proposal samples and associated log probability have already been calculated
        # using the stored model. These form a normalized probability distribution.
        log_prob_proposal = self.samples["log_prob"].to_numpy()

        delta_log_prob_target = np.zeros(len(self.samples))
        if "delta_log_prob_target" in self.samples.columns:
            delta_log_prob_target = self.samples["delta_log_prob_target"].to_numpy()

        # select parameters in self.samples (required as log_prob and potentially gnpe
        # proxies are also stored in self.samples, but are not needed for the likelihood.
        # TODO: replace by self.metadata["train_settings"]["data"]["inference_parameters"]
        param_keys = [k for k, v in self.prior.items() if not isinstance(v, Constraint)]
        aux_keys = list(set(self.samples.keys()).difference(param_keys))
        theta = self.samples[param_keys]
        aux_params = self.samples[aux_keys]

        # Calculate the (un-normalized) target density as prior times likelihood,
        # evaluated at the same sample points.
        log_prior = self.prior.ln_prob(theta, axis=0)

        # Check whether any constraints are violated that involve parameters not
        # already present in theta. NOTE what is the point of this won't the samples already be in the prior??
        # constraints = self.prior.evaluate_constraints(theta)
        constraints = np.array([self.prior.evaluate_constraints(t) for _, t in theta.iterrows()])
        np.putmask(log_prior, constraints == 0, -np.inf)

        # The prior may evaluate to -inf for certain samples. For these, we do not want
        # to evaluate the likelihood, in particular because it may not even be possible
        # to generate data outside the prior (e.g., for BH spins > 1). Since there is
        # no point in keeping these samples, we simply drop them; this means we do not
        # have to make special exceptions for outside-prior samples elsewhere in the
        # code. They do not contribute directly to the evidence or the effective sample
        # size, so we are not losing anything useful. However, it is important to count
        # them in num_samples when computing the evidence, since they contribute to the
        # normalization of the proposal distribution.

        within_prior = (log_prior + delta_log_prob_target) != -np.inf
        num_samples = len(self.samples)
        if num_samples != np.sum(within_prior):
            print(
                f"Of {num_samples} samples, "
                f"{num_samples - np.sum(within_prior)} lie outside the prior. "
                f"Dropping these."
            )
            theta = theta.iloc[within_prior].reset_index(drop=True)
            aux_params = aux_params.iloc[within_prior].reset_index(drop=True)
            log_prob_proposal = log_prob_proposal[within_prior]
            log_prior = log_prior[within_prior]
            if delta_log_prob_target != 0:
                delta_log_prob_target = delta_log_prob_target[within_prior]

        print(f"Calculating {len(theta)} likelihoods.")
        t0 = time.time()
        log_likelihood = self.likelihood.log_likelihood_multi(
            theta, num_processes=num_processes
        )
        print(f"Done. This took {time.time() - t0:.2f} seconds.")

        # Calculate weights and normalize them to have mean 1.
        log_weights = (
            log_prior + log_likelihood + delta_log_prob_target - log_prob_proposal
        )
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= np.mean(weights)

        self.samples = theta
        self.samples["log_prob"] = log_prob_proposal  # Proposal log_prob, not target!
        self.samples["weights"] = weights
        self.samples["log_likelihood"] = log_likelihood
        self.samples["log_prior"] = log_prior
        for k in aux_keys:
            self.samples[k] = aux_params[k]
        # self.samples["delta_log_prob_target"] = delta_log_prob_target

        # The evidence
        #           Z = \int d\theta \pi(\theta) L(\theta),
        #
        #                   where   \pi = prior,
        #                           L = likelihood.
        #
        # For importance sampling, we estimate this using Monte Carlo integration using
        # the proposal distribution q(\theta),
        #
        #           Z = \int d\theta q(\theta) \pi(\theta) L(\theta) / q(\theta)
        #             ~ (1/N) \sum_i \pi(\theta_i) L(\theta_i) / q(\theta_i)
        #
        #                   where we are summing over samples \theta_i ~ q(\theta).
        #
        # The integrand is just the importance weight (prior to any normalization). It
        # is more numerically stable to evaluate log(Z),
        #
        #           log Z ~ \log \sum_i \exp( log \pi_i + log L_i - log q_i ) - log N
        #                 = logsumexp ( log_weights ) - log N
        #
        # Notes
        # -----
        #   * We use the logsumexp functions, which is more numerically stable.
        #   * N = num_samples is the *original* number of samples (including the
        #     zero-weight ones that we dropped).
        #   * q, \pi, L must be distributions in the same parameter space (the same
        #     coordinates). We have undone any standardizations so this is the case.

        self.n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
        # ESS computed with len(weights) in denominator instead of num_samples,
        # since we are interested in ESS per *likelihood evaluation*, not per
        # Dingo sample.
        self.effective_sample_size = self.n_eff / len(weights)

        self.log_evidence = logsumexp(log_weights) - np.log(num_samples)
        log_weights_all = np.pad(
            log_weights - self.log_evidence,
            (num_samples - len(log_weights), 0),
            constant_values=-np.inf,
        )
        assert np.allclose(np.mean(np.exp(log_weights_all)), 1)
        # log_evidence_std = 1/sqrt(n) (evidence_std / evidence)
        self.log_evidence_std = np.sqrt(
            (num_samples - self.n_eff) / (num_samples * self.n_eff)
        )

    def write_pesummary(self, filename):
        from pesummary.io import write
        from pesummary.utils.samples_dict import SamplesDict

        samples_dict = SamplesDict(self.samples)
        write(
            samples_dict,
            package=self._pesummary_package,
            file_format="hdf5",
            filename=filename,
        )
        # TODO: Save much more information.

    def to_result(self) -> Result:
        """
        Export samples, metadata, and context information to a Result instance,
        which can be used for saving or, e.g., importance sampling, training an
        unconditional flow, etc.

        Returns
        -------
        Result
        """
        data_dict = {k: getattr(self, k, None) for k in RESULT_DATA_KEYS}
        # *COPY* the metadata to avoid recursion errors when creating new objects.
        data_dict["settings"] = copy.deepcopy(self.metadata)
        return self._result_class(dictionary=data_dict)

    def to_hdf5(self, label="", outdir="."):
        dataset = self.to_result()
        file_name = label + "_result.hdf5"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        dataset.to_file(file_name=Path(outdir, file_name))


class GNPESampler(Sampler):
    """
    Base class for GNPE sampler. It wraps a PosteriorModel *and* a standard Sampler for
    initialization. The former is used to generate initial samples for Gibbs sampling.

    A GNPE network is conditioned on additional "proxy" context theta^, i.e.,

    p(theta | theta^, d)

    The theta^ depend on theta via a fixed kernel p(theta^ | theta). Combining these
    known distributions, this class uses Gibbs sampling to draw samples from the joint
    distribution,

    p(theta, theta^ | d)

    The advantage of this approach is that we are allowed to perform any transformation of
    d that depends on theta^. In particular, we can use this freedom to simplify the
    data, e.g., by aligning data to have merger times = 0 in each detector. The merger
    times are unknown quantities that must be inferred jointly with all other
    parameters, and GNPE provides a means to do this iteratively. See
    https://arxiv.org/abs/2111.13139 for additional details.

    Gibbs sampling breaks access to the probability density, so this must be recovered
    through other means. One way is to train an unconditional flow to represent p(theta^
    | d) for fixed d based on the samples produced through the GNPE Gibbs sampling.
    Starting from these, a single Gibbs iteration gives theta from the GNPE network,
    along with the probability density in the joint space. This is implemented in
    GNPESampler provided the init_sampler provides proxies directly and num_iterations
    = 1.

    Attributes (beyond those of Sampler)
    ----------
    init_sampler : Sampler
        Used for providing initial samples for Gibbs sampling.
    num_iterations : int
        Number of Gibbs iterations to perform.
    iteration_tracker : IterationTracker  **not set up**
    remove_init_outliers : float  **not set up**
    """

    def __init__(
        self,
        model: PosteriorModel,
        init_sampler: Sampler,
        num_iterations: int = 1,
    ):
        """
        Parameters
        ----------
        model : PosteriorModel
        init_sampler : Sampler
            Used for generating initial samples
        num_iterations : int
            Number of GNPE iterations to be performed by sampler.
        """
        self.gnpe_parameters = []  # Should be set in subclass _initialize_transform()

        super().__init__(model)
        self.init_sampler = init_sampler
        self.num_iterations = num_iterations
        self.iteration_tracker = None
        # remove self.remove_init_outliers of lowest log_prob init samples before gnpe
        self.remove_init_outliers = 0.0

    @property
    def init_sampler(self):
        return self._init_sampler

    @init_sampler.setter
    def init_sampler(self, value):
        self._init_sampler = value
        # Copy this so it persists if we delete the init model.
        self.metadata["init_model"] = self._init_sampler.model.metadata.copy()
        if self._init_sampler.unconditional_model:
            self.context = self._init_sampler.context
            self.event_metadata = self._init_sampler.event_metadata

    @property
    def num_iterations(self):
        """The number of GNPE iterations to perform when sampling."""
        return self._num_iterations

    @num_iterations.setter
    def num_iterations(self, value):
        self._num_iterations = value
        self.metadata["num_iterations"] = self._num_iterations

    @property
    def gnpe_proxy_parameters(self):
        return [p + "_proxy" for p in self.gnpe_parameters]

    def _kernel_log_prob(self, samples):
        raise NotImplementedError("To be implemented in subclass.")

    def _run_sampler(
        self,
        num_samples: int,
        context: Optional[dict] = None,
    ) -> dict:
        if context is None:
            raise ValueError("self.context must be set to run sampler.")

        data_ = self.init_sampler.transform_pre(context)

        # TODO: Reimplement outlier removal in IterationTracker? Save setting somewhere.
        if self.remove_init_outliers == 0.0:
            init_samples = self.init_sampler._run_sampler(num_samples, context)
        else:
            if self.num_iterations == 1:
                print(
                    f"Warning: Removing initial outliers, but only carrying out "
                    f"{self.num_iterations} GNPE iteration. This risks biasing "
                    f"results."
                )
            init_samples = self.init_sampler._run_sampler(
                math.ceil(num_samples / (1 - self.remove_init_outliers)), context
            )
            thr = torch.quantile(init_samples["log_prob"], self.remove_init_outliers)
            inds = torch.where(init_samples["log_prob"] >= thr)[0][:num_samples]
            init_samples = {k: v[inds] for k, v in init_samples.items()}

        # We could be starting with either the GNPE parameters *or* their proxies,
        # depending on the nature of the initialization network.

        start_with_proxies = False
        proxy_log_prob = None
        proxies = {}

        if {p + "_proxy" for p in self.gnpe_parameters}.issubset(init_samples.keys()):
            start_with_proxies = True
            proxy_log_prob = init_samples["log_prob"]
            proxies = {k: init_samples[k] for k in self.gnpe_proxy_parameters}
            # proxies is a dict of torch.Tensors, since it came from _run_sampler(),
            # not run_sampler(). Clone it for a later assertion check.
            init_proxies = {k: v.clone() for k, v in proxies.items()}

        x = {"extrinsic_parameters": init_samples, "parameters": {}}

        #
        # Gibbs sample.
        #

        # Saving as an attribute so we can later access the intermediate Gibbs samples.
        self.iteration_tracker = IterationTracker(store_data=True)

        for i in range(self.num_iterations):
            start_time = time.time()

            if start_with_proxies and i == 0:
                x["extrinsic_parameters"] = proxies.copy()
            else:
                x["extrinsic_parameters"] = {
                    k: x["extrinsic_parameters"][k] for k in self.gnpe_parameters
                }

            # TODO: Depending on whether start_with_proxies is True, this might end up
            #  comparing proxies vs gnpe_parameters for the first iteration.
            self.iteration_tracker.update(
                {k: v.cpu().numpy() for k, v in x["extrinsic_parameters"].items()}
            )

            d = data_.clone()
            x["data"] = d.expand(num_samples, *d.shape)

            x = self.transform_pre(x)

            time_sample_start = time.time()
            self.model.model.eval()
            with torch.no_grad():
                y, log_prob = self.model.model.sample_and_log_prob(
                    x["data"], x["context_parameters"]
                )
            time_sample_end = time.time()

            x["parameters"] = y
            x["log_prob"] = log_prob
            x = self.transform_post(x)

            # Extract the proxy parameters from x["extrinsic_parameters"]. These have
            # not been standardized. They are persistent from before sampling took place,
            # since this is when they were placed here and their values should not have
            # changed.
            proxies = {
                p: x["extrinsic_parameters"][p] for p in self.gnpe_proxy_parameters
            }

            print(
                f"it {i}.\tmin pvalue: {self.iteration_tracker.pvalue_min:.3f}"
                f"\tproxy mean: ",
                *[f"{torch.mean(v).item():.5f}" for v in proxies.values()],
                "\tproxy std:",
                *[f"{torch.std(v).item():.5f}" for v in proxies.values()],
                "\ttimes:",
                time_sample_start - start_time,
                time_sample_end - time_sample_start,
                time.time() - time_sample_end,
            )

        #
        # Prepare final result.
        #

        if start_with_proxies and self.num_iterations == 1:
            # In this case it makes sense to save the log_prob and the proxy parameters.

            samples = x["parameters"]
            samples["log_prob"] = x["log_prob"] + proxy_log_prob

            # The log_prob returned by gnpe is not just the log_prob over parameters
            # theta, but instead the log_prob in the *joint* space q(theta,theta^|x),
            # including the proxies theta^. For importance sampling this means,
            # that the target density is
            #
            #       p(theta,theta^|x) = p(theta^|theta) * p(theta|x).
            #
            # We compute log[p(theta^|theta)] below and store it as
            # samples["delta_log_prob_target"], such that for importance sampling we
            # only need to evaluate log[p(theta|x)] and add this correction.

            # Proxies should be sitting in extrinsic_parameters.
            all_params = {**x["extrinsic_parameters"], **samples}
            all_params = {k: torch_detach_to_cpu(v) for k, v in all_params.items()}
            kernel_log_prob = self._kernel_log_prob(all_params)
            samples["delta_log_prob_target"] = torch.Tensor(kernel_log_prob)

        else:
            # Otherwise we only save the inference parameters, and no log_prob.
            # Alternatively we could save the entire chain and the log_prob, but this
            # is not useful for our purposes.

            samples = x["parameters"]

        # Include the proxies along with the inference parameters. The variable proxies
        # gets set at the end of each Gibbs loop.
        samples.update(proxies)

        # Safety check for unconditional flows. Make sure the proxies haven't changed.
        if start_with_proxies and self.num_iterations == 1:
            for k in proxies:
                assert torch.equal(proxies[k], init_proxies[k])

        return samples