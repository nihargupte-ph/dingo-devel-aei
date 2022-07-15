data {
  int nobs;
  int nsamp;

  real chis[nobs, nsamp];
}

/* Things will be smoother if we KDE over the data; this computes a Scott-like
/* rule for the bandwidth of that KDE. */
transformed data {
  real bws[nobs];

  for (i in 1:nobs) {
    bws[i] = sd(chis[i,:])/nsamp^(1.0/5.0); /* Scott's rule for KDE. */
  }
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
  /* Diffuse, unit-scale priors on both mu and sigma to ensure posterior is
     proper; if chi is not intrinsically unit-scale it should be transformed
     before fitting here. */

  mu ~ normal(0, 1);
  sigma ~ normal(0,1);

  /* Here we use a trick to analytically marginalize over the KDE likelihood for
     each systems true chi value: the product of N(mu, sigma) prior and KDE
     likelihood is just a Gaussian, so integrating over chi is just convolving
     the two Gaussians together, and yields yet another Gaussian, with variances
     summed. */
  for (i in 1:nobs) {
    real logps[nsamp];
    real sigma_i = sqrt(sigma*sigma + bws[i]*bws[i]);
    for (j in 1:nsamp) {
      logps[j] = normal_lpdf(chis[i,j] | mu, sigma_i);
    }
    target += log_sum_exp(logps) - log(nsamp);
  }
}

generated quantities {
  /* We draw a sample from the Normal population of chi values at the value of
  /* mu and sigma we have fit. */
  real pop;

  pop = normal_rng(mu, sigma);
}
