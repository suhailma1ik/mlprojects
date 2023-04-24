import numpy as np
import matplotlib.pyplot as plt

# Set the mean and standard deviation of the normal distribution
mu, sigma = 0, 1

# Generate 1000 random samples from the normal distribution
samples = np.random.normal(mu, sigma, 1000)

# Plot the histogram of the samples
plt.hist(samples, bins=50, density=True)

# Plot the probability density function of the normal distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sigma**2))
plt.plot(x, p, 'k', linewidth=2)

# Set the plot labels and title
plt.xlabel('X-axis')
plt.ylabel('Probability density')
plt.title('Gaussian distribution')

# Show the plot
plt.show()
