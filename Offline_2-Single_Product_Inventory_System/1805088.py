import math
import numpy as np
from queue import Queue
import os

INPUT_FILE_DIR = "./in.txt"
OUTPUT_FILE_DIR = "./out.txt"

NONE = 0
ORDER_ARRIVAL = 1
DEMAND = 2
END = 3
EVALUATE = 4

INFINITE = 1.0e+30

class PMMLCG:
    MODLUS = 2147483647
    MULT1 = 24112
    MULT2 = 26143

    def __init__(self):
        self.zrng = [
            1, 1973272912, 281629770, 20006270, 1280689831, 2096730329, 1933576050,
            913566091, 246780520, 1363774876, 604901985, 1511192140, 1259851944,
            824064364, 150493284, 242708531, 75253171, 1964472944, 1202299975,
            233217322, 1911216000, 726370533, 403498145, 993232223, 1103205531,
            762430696, 1922803170, 1385516923, 76271663, 413682397, 726466604,
            336157058, 1432650381, 1120463904, 595778810, 877722890, 1046574445,
            68911991, 2088367019, 748545416, 622401386, 2122378830, 640690903,
            1774806513, 2132545692, 2079249579, 78130110, 852776735, 1187867272,
            1351423507, 1645973084, 1997049139, 922510944, 2045512870, 898585771,
            243649545, 1004818771, 773686062, 403188473, 372279877, 1901633463,
            498067494, 2087759558, 493157915, 597104727, 1530940798, 1814496276,
            536444882, 1663153658, 855503735, 67784357, 1432404475, 619691088,
            119025595, 880802310, 176192644, 1116780070, 277854671, 1366580350,
            1142483975, 2026948561, 1053920743, 786262391, 1792203830, 1494667770,
            1923011392, 1433700034, 1244184613, 1147297105, 539712780, 1545929719,
            190641742, 1645390429, 264907697, 620389253, 1502074852, 927711160,
            364849192, 2049576050, 638580085, 547070247
        ]

    def generate(self, stream):
        zi = self.zrng[stream]
        lowprd = (zi & 65535) * self.MULT1
        hi31 = (zi >> 16) * self.MULT1 + (lowprd >> 16)
        zi = ((lowprd & 65535) - self.MODLUS) + ((hi31 & 32767) << 16) + (hi31 >> 15)
        if zi < 0:
            zi += self.MODLUS
        lowprd = (zi & 65535) * self.MULT2
        hi31 = (zi >> 16) * self.MULT2 + (lowprd >> 16)
        zi = ((lowprd & 65535) - self.MODLUS) + ((hi31 & 32767) << 16) + (hi31 >> 15)
        if zi < 0:
            zi += self.MODLUS
        self.zrng[stream] = zi
        return (zi >> 7 | 1) / 16777216.0

    def set_seed(self, zset, stream):
        self.zrng[stream] = zset

    def get_seed(self, stream):
        return self.zrng[stream]


# Create Single Product Inventory System
class SPIS:
    def __init__(self, input_file_path, output_file_path, num_of_events = 4):
        with open(input_file_path, "r") as input_file:
            input = input_file.readline()
            self.initial_inventory_level, self.num_of_months, self.num_of_policies = map(int, input.split(' '))
            
            input = input_file.readline()
            self.num_of_demand_sizes, self.mean_inter_demand = int(input.split(' ')[0]), float(input.split(' ')[1])
            
            input = input_file.readline()
            self.setup_cost, self.per_unit_incremental_cost, self.holding_cost, self.storage_cost = map(float, input.split(' '))
            
            input = input_file.readline()
            self.min_lag, self.max_lag = map(float, input.split(' '))
            
            input = input_file.readline()
            self.cum_prob_of_sequential_demand = list(map(float, input.split(' ')))
            
            self.policies = []
            for policies in range(self.num_of_policies):
                input = input_file.readline()
                self.policies.append(list(map(int, input.split(' '))))
                
            self.output_file = open(output_file_path, "a+")
            self.num_of_events = num_of_events
            
            # Initialize hyperparameters
            self.amount = 0
            self.bigs = 0
            self.inventory_level = 0
            self.next_event_type = NONE        
            self.smalls = 0
            
            self.area_holding = 0.0
            self.area_shortage = 0.0
            self.simulation_time = 0.0
            self.time_of_last_event = 0.0
            self.time_next_event = [0.0] * (self.num_of_events + 1)
            self.total_ordering_cost = 0.0
                        
            self.prime_mod_generator = PMMLCG()
            
    def reportInputParams(self):
        self.output_file.write(f"------Single-Product Inventory System------\n\n")
        self.output_file.write(f"Initial inventory level: {self.initial_inventory_level} items\n\n")
        self.output_file.write(f"Number of demand sizes: {self.num_of_demand_sizes}\n\n")
        self.output_file.write(f"Distribution function of demand sizes: ")
        for i in range(self.num_of_demand_sizes):
            self.output_file.write(f"{self.cum_prob_of_sequential_demand[i]:.2f} ")
        self.output_file.write("\n\n")
        self.output_file.write(f"Mean inter-demand time: {self.mean_inter_demand:.2f} months\n\n")
        self.output_file.write(f"Delivery lag range: {self.min_lag:.2f} to {self.max_lag:.2f} months\n\n")
        self.output_file.write(f"Length of simulation: {self.num_of_months} months\n\n")
        self.output_file.write("Costs:\n")
        self.output_file.write(f"K = {self.setup_cost:.2f}\n")
        self.output_file.write(f"i = {self.per_unit_incremental_cost:.2f}\n")
        self.output_file.write(f"h = {self.holding_cost:.2f}\n")
        self.output_file.write(f"pi = {self.storage_cost:.2f}\n\n")
        self.output_file.write(f"Number of policies: {self.num_of_policies}\n\n")
        self.output_file.write(f"Policies:\n")
        self.output_file.write(f"--------------------------------------------------------------------------------------------------\n")
        self.output_file.write(f" Policy        Avg_total_cost     Avg_ordering_cost      Avg_holding_cost     Avg_shortage_cost\n")
        self.output_file.write(f"--------------------------------------------------------------------------------------------------\n\n")
    
    def exponen(self, exponential_probability_distribution_mean):
        # return -1 * exponential_probability_distribution_mean * math.log(round(self.prime_mod_generator.generate(1), 6))
        return -1 * exponential_probability_distribution_mean * math.log(self.prime_mod_generator.generate(1))
            
    def initialize_simulation(self):
        # Initialize the simulation clock
        self.simulation_time = 0.0
        
        # Initialize the state variables
        self.inventory_level = self.initial_inventory_level
        self.time_of_last_event = 0.0
        
        # Initialize the statistical counters
        self.total_ordering_cost = 0.0
        self.area_holding = 0.0
        self.area_shortage = 0.0
        
        # Initialize the event list. Since no order is outstanding, the orderarrival 
        # event is eliminated from consideration
        self.time_next_event[ORDER_ARRIVAL] = INFINITE
        self.time_next_event[DEMAND] = self.simulation_time + self.exponen(self.mean_inter_demand)
        self.time_next_event[EVALUATE] = 0.0
        self.time_next_event[END] = self.num_of_months
        
    def timing(self):        
        min_next_event_time = 1.0e+29
        self.next_event_type = NONE
        
        for event_no in range(1, self.num_of_events+1):
            if(self.time_next_event[event_no] < min_next_event_time):
                min_next_event_time = self.time_next_event[event_no]
                self.next_event_type = event_no
         
        if(self.next_event_type == NONE):
            print("No event left in Event List!!")
            exit()
                   
        self.simulation_time = min_next_event_time
        
    def update_time_avg_stats(self):     
        # Determine the status of the inventory level during the previous interval.
        # If the inventory level during the previous interval was negative, update
        # area_shortage. If it was positive, update area_holding. If it was zero,
        # no update is needed.   
        if (self.inventory_level < 0):
            self.area_shortage -= self.inventory_level * (self.simulation_time - self.time_of_last_event)
        else:
            self.area_holding += self.inventory_level * (self.simulation_time - self.time_of_last_event)
        
        self.time_of_last_event = self.simulation_time
        
    def order_arrival(self):
        # Increment the inventory level by the amount ordered
        self.inventory_level += self.amount
        
        # Since no order is now outstanding, eliminate the 
        # order-arrival event from consideration
        self.time_next_event[ORDER_ARRIVAL] = INFINITE
    
    def random_integer(self, probability_distributions):
        u = self.prime_mod_generator.generate(1)
        i = 0
        while u >= probability_distributions[i]:
            i = i + 1
        return i + 1
        
    def demand(self):
        # Decrement the inventory level by a generated demand size
        self.inventory_level -= self.random_integer(self.cum_prob_of_sequential_demand)

        # Schedule the time of the next demand
        self.time_next_event[DEMAND] = self.simulation_time + self.exponen(self.mean_inter_demand)
    
    def uniform(self, a, b):
        # Return a U(a,b) random variate
        return a + self.prime_mod_generator.generate(1) * (b - a)
       
    def evaluate(self):
        # Check whether the inventory level is less than smalls
        if self.inventory_level < self.smalls:
            # The inventory level is less than smalls, so place 
            # an order for the appropriate amount
            self.amount = self.bigs - self.inventory_level
            self.total_ordering_cost += self.setup_cost + self.per_unit_incremental_cost * self.amount
            
            # Schedule the arrival of the order
            self.time_next_event[ORDER_ARRIVAL] = self.simulation_time + self.uniform(self.min_lag, self.max_lag)

        # Regardless of the place-order decision, schedule 
        # the next inventory evaluation
        self.time_next_event[EVALUATE] = self.simulation_time + 1.0
        
    def report(self):
        # Compute and write estimates of desired measures of performance. */
        avg_ordering_cost = self.total_ordering_cost / self.num_of_months
        avg_holding_cost = self.holding_cost * self.area_holding / self.num_of_months
        avg_shortage_cost = self.storage_cost * self.area_shortage / self.num_of_months
        self.output_file.write("(%2d,%3d) %19.2f %19.2f %19.2f %19.2f\n\n" % (self.smalls, self.bigs, avg_ordering_cost + avg_holding_cost + avg_shortage_cost, avg_ordering_cost, avg_holding_cost, avg_shortage_cost))
        
    def simulation(self, policy):
        # Read the inventory policy, and initialize the simulation
        self.smalls = policy[0]
        self.bigs = policy[1]
        self.initialize_simulation()
        
        # Run the simulation until it terminates after an end-simulation event
        # (type 3) occurs
        while True:
            # Determine the next event
            self.timing()
            
            # Update time-average statistical accumulators
            self.update_time_avg_stats()
            
            # Invoke the appropriate event function
            if self.next_event_type == ORDER_ARRIVAL:
                # print("Arrival")
                self.order_arrival()
            elif self.next_event_type == DEMAND:
                # print("Demand")
                self.demand()
            elif self.next_event_type == EVALUATE:
                # print("Evaluate")
                self.evaluate()
            elif self.next_event_type == END:
                # print("End")
                self.report()
                break    
            
def main():
    inventory_system = SPIS(INPUT_FILE_DIR, OUTPUT_FILE_DIR)
    inventory_system.reportInputParams()
    policies = inventory_system.policies
    for i in range(inventory_system.num_of_policies):
        inventory_system.simulation(policies[i])
    inventory_system.output_file.write(f"--------------------------------------------------------------------------------------------------")      
    
if __name__ == "__main__":
    main()