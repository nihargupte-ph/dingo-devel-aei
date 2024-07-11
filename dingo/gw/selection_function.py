
def selection_alpha(self, hyper_injection_parameters, num_monte_carlo_samples=1000):
    """
    Compute the selection function

    \int p_det(\theta) p(\theta | \lambda) d\theta

    We compute this by monte carlo sampling p(\theta | \lambda)
    """
    num_monte_carlo_samples = 1000
    injection_parameters = self.sample_injection_parameters(hyper_injection_parameters, num_injections=num_monte_carlo_samples)
    p_det_values = self.p_det(injection_parameters)
    alpha = jnp.sum(p_det_values) / num_monte_carlo_samples

    # TODO return uncertainty of monte carlo estimate 
    return alpha