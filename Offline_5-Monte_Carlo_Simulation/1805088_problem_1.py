import numpy as np

OUTPUT_FILE_DIR = "./fission_output.txt"

class NuclearChainReactionSimulator:
    def __init__(self, generations=10, trials=10000):
        self.generations = generations
        self.trials = trials
        # Initializing probabilities
        self.probabilities = self._calculate_probabilities()
        # Initialize results matrix
        self.results = np.zeros((generations, 5), dtype=int)  # For storing counts of 0-4 neutrons
        self.output_file = open(OUTPUT_FILE_DIR, "a+")

    def _calculate_probabilities(self):
        # Calculate probabilities for 0-3 new neutrons, ensuring they sum to 1
        prob_1_to_3 = [(0.2126) * (0.5893) ** (i - 1) for i in range(1, 4)]
        prob_0 = 1 - sum(prob_1_to_3)  # Probability for 0 new neutrons
        return [prob_0] + prob_1_to_3

    def _simulate_generation(self, current_neutrons):
        # Simulate the neutron generation process
        next_gen_neutrons = 0
        for _ in range(current_neutrons):
            # Choose how many neutrons are produced by this neutron
            produced = np.random.choice([0, 1, 2, 3], p=self.probabilities)
            next_gen_neutrons += produced
        return next_gen_neutrons

    def run_simulation(self):
        for _ in range(self.trials):
            current_neutrons = 1  # Start with one neutron
            for gen in range(self.generations):
                current_neutrons = self._simulate_generation(current_neutrons)
                # Record the result, ensuring it fits into the 0-4 range
                self.results[gen, min(current_neutrons, 4)] += 1

    def calculate_probabilities(self):
        # Convert counts to probabilities
        return self.results / self.trials

    def display_results(self):
        probabilities = self.calculate_probabilities()
        for gen in range(self.generations):
            # print(f"Generation {gen + 1}: {probabilities[gen]}")
            self.output_file.write(f"Generation-{gen + 1}:\n")
            for i in range(0, 5):
                self.output_file.write(f"p[{i}] = {probabilities[gen][i]}\n")
                
            self.output_file.write(f"\n")

# Create an instance of the simulator and run it
simulator = NuclearChainReactionSimulator()
simulator.run_simulation()
simulator.display_results()
