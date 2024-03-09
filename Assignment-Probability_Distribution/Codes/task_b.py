import numpy as np
from scipy.stats import gamma, weibull_min

# Load your data
with open('sample.txt', 'r') as file:
    data = np.loadtxt(file)

# Gamma Distribution: MLE
# The gamma distribution parameters can be estimated with the `fit` method, which uses MLE under the hood
alpha_hat, loc_hat, beta_hat = gamma.fit(data, floc=0)  # Keeping location parameter fixed at 0

# Weibull Distribution: MLE
# Similar to the gamma distribution, we can estimate the parameters of the Weibull distribution using the `fit` method
params = weibull_min.fit(data, floc=0)  # Keeping location parameter fixed at 0
c_hat, loc_hat, scale_hat = params

print("Gamma Distribution Parameters (alpha, loc, beta):", alpha_hat, loc_hat, beta_hat)
print("Weibull Distribution Parameters (c, loc, scale):", c_hat, loc_hat, scale_hat)
