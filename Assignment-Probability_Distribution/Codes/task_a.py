import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Read data from file
with open('sample.txt', 'r') as file:
    data = np.loadtxt(file)

#calculate mean, median and mode
mean = np.mean(data)
median = np.median(data)
std = np.std(data)
skewness = stats.skew(data)

print('Mean: ', mean)
print('Median: ', median)
print('Standard Deviation: ', std)
print('Skewness: ', skewness)

# -------------------- 2d Plot -------------------- #
# Plot the histogram of the data
plt.hist(data, bins=100, density=True, alpha=0.6, color='g')

# Overlay a normal distribution for comparison
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title('Data Distribution with Normal Distribution Overlay')
plt.xlabel('Data values')
plt.ylabel('Density')
plt.show()


# -------------------- 3d Plot -------------------- #
# # Create histogram bins
# hist, bins = np.histogram(data, bins=50)

# # Width of each bin
# width = np.diff(bins)

# # Initialize the plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Create the 3D histogram
# for i in range(len(hist)):
#     ax.bar3d(x=bins[i], y=0, z=0, dx=width[i], dy=2, dz=hist[i])

# # Set labels and title
# ax.set_xlabel('Data values')
# ax.set_ylabel('Y')
# ax.set_zlabel('Count')
# ax.set_title('3D Histogram of Data')

# # Show the plot
# plt.show()