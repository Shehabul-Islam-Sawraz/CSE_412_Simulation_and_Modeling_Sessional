import numpy as np
import matplotlib.pyplot as plt

def simulate_secretary_problem(n, s, m, iterations=10000):
    successes = 0
    for _ in range(iterations):
        # Generate a random permutation of candidates
        candidates = np.random.permutation(n) + 1

        # Set the standard as the best among the first m candidates
        standard = max(candidates[:m]) if m > 0 else 0

        selected = False
        # Go through the rest of the candidates
        for candidate in candidates[m:]:
            if candidate > standard:  # Select the first candidate better than the standard
                # Check if the selected candidate meets the success criteria
                if candidate > n - s:
                    successes += 1
                selected = True
                break
                
        if selected==False:
            if n - candidates[-1] < s:
                successes += 1
    return successes / iterations

# Population size
n = 100

# Success criteria
success_criteria = [1, 3, 5, 10]

# Sample sizes to test
sample_sizes = range(n)

# Number of iterations for each simulation
iterations = 10000

# Store results
results = {s: [] for s in success_criteria}

# Simulate for each success criteria and sample size
for s in success_criteria:
    for m in sample_sizes:
        success_rate = simulate_secretary_problem(n, s, m, iterations)
        results[s].append(success_rate)

# Plotting
plt.figure(figsize=(10, 6))
for s in success_criteria:
    plt.plot(sample_sizes, results[s], label=f'Top {s}')

plt.xlabel('Sample Size (m)')
plt.ylabel('Success Rate')
plt.title('Success Rate vs. Sample Size in Secretary Problem')
plt.legend()
plt.grid(True)
plt.show()
