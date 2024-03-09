import numpy as np
from scipy.stats import chi2, gamma

# Load the dataset
with open('sample.txt', 'r') as file:
    data = np.loadtxt(file)

# Estimate the parameters of the gamma distribution (shape and scale) for the given data
alpha_hat, loc_hat, scale_hat = gamma.fit(data, floc=0)

# Define the number of bins for the chi-square test
num_bins = 50

# Create the bins for the histogram
bin_edges = np.linspace(min(data), max(data), num_bins+1)
observed_freq, _ = np.histogram(data, bins=bin_edges)

# Generate the expected frequencies for each bin
expected_freq = np.zeros(num_bins)
for i in range(num_bins):
    expected_freq[i] = gamma.cdf(bin_edges[i+1], alpha_hat, scale=scale_hat) - gamma.cdf(bin_edges[i], alpha_hat, scale=scale_hat)

# The expected frequencies need to be scaled to the total number of observations
expected_freq *= len(data)

# Perform the Chi-Square Goodness of Fit Test
chi_square_statistic = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()

# # Determine the degrees of freedom
# degrees_of_freedom = num_bins - 1 - 2  # Two parameters estimated from the data

# # Set the significance level
# alpha = 0.05

# # Determine the critical value
# critical_value = chi2.ppf(1 - alpha, degrees_of_freedom)

p_value = chi2.sf(chi_square_statistic, num_bins - 1 - 2)  # Degrees of freedom = number of bins - 1 - number of estimated parameters

print(f"Chi-squared statistic: {chi_square_statistic}")
print(f"p value: {p_value}")

if p_value < 0.05:
    print("Reject the null hypothesis - The distribution does not fit the data well.")
else:
    print("Fail to reject the null hypothesis - The distribution fits the data well.")