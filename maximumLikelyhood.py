import numpy as np
from scipy.stats import norm

def maximum_likelihood_estimation(data):
    # Compute the sample mean and standard deviation
    mu = np.mean(data)
    sigma = np.std(data)
    
    # Define the likelihood function for a normal distribution
    def likelihood(x):
        return np.prod(norm.pdf(x, loc=mu, scale=sigma))
    
    # Maximize the likelihood function using scipy's minimize function
    from scipy.optimize import minimize
    result = minimize(lambda x: -likelihood(x), x0=[mu, sigma])
    
    # Return the maximum likelihood estimates of the parameters
    return result.x


# implementation
data = np.random.normal(loc=5, scale=2, size=100)

# Estimate the parameters of the normal distribution using MLE
mu, sigma = maximum_likelihood_estimation(data)
print('mu =', mu)
print('sigma =', sigma)