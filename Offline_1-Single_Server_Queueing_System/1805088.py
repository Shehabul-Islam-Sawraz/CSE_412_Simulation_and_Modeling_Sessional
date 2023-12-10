import math
import numpy as np
from queue import Queue
import os

INPUT_FILE_DIR = "./input_file/in.txt"
OUTPUT_FILE_DIR = "./output_files/"
RESULT_FILE = "results.txt"
EVENT_ORDERS_FILE = "event_orders.txt"

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
    
    
pmmlcg = PMMLCG()
def exponen(exponential_probability_distribution_mean):
    return -1 * exponential_probability_distribution_mean * math.log(round(pmmlcg.generate(1), 6))

def initialize_simulation(mean_inter_arrival_time):
    global simulation_time, server_status, number_in_queue, \
        time_of_last_event, num_customers_delayed, total_delays, \
        area_num_in_queue, area_server_status, \
        total_customer_arrived, time_next_event_arrival, \
        time_next_event_departure, num_of_event, num_of_arrival, \
        num_of_departure, times_of_arrival
    
    # System States
    server_status = False
    number_in_queue = 0
    time_of_last_event = 0
    total_customer_arrived = 1
    num_of_event = 0
    num_of_arrival = 0
    num_of_departure = 0
    
    # Simulation States
    simulation_time = 0
    
    # Event States
    time_next_event_arrival = exponen(mean_inter_arrival_time)
    time_next_event_departure = math.inf
    times_of_arrival = Queue()
    
    # Statistical Values
    num_customers_delayed = 0
    total_delays = 0
    area_num_in_queue = 0
    area_server_status = 0
    
def inc_num_customer_delayed():
    global num_customers_delayed
    
    num_customers_delayed += 1
    with open(OUTPUT_FILE_DIR+EVENT_ORDERS_FILE, "a+") as event:
        event.write(
            f'\n---------No. of customers delayed: {num_customers_delayed}--------\n\n'
        )
        
def update_time_avg_stats():
    global area_num_in_queue, area_server_status,\
        simulation_time, time_of_last_event
        
    area_num_in_queue += number_in_queue * (simulation_time - time_of_last_event)
    area_server_status += server_status * (simulation_time - time_of_last_event)
    time_of_last_event = simulation_time
    
def timing():
    global simulation_time
    
    if (time_next_event_arrival < time_next_event_departure):
       simulation_time = time_next_event_arrival
       return 1 # 1 = arrival event
    else:
        simulation_time = time_next_event_departure
        return 2 # 2 = departure event
    
def arrive(mean_inter_arrival_time, mean_service_time, num_of_delays_required):
    global num_of_arrival, time_next_event_arrival, \
        total_customer_arrived, number_in_queue, \
        area_num_in_queue, area_server_status, \
        num_customers_delayed, server_status, \
        time_next_event_departure, time_of_last_event, \
        times_of_arrival
        
    num_of_arrival += 1
    with open(OUTPUT_FILE_DIR+EVENT_ORDERS_FILE, "a+") as event:
        event.write(
            f'{num_of_event}. Next event: Customer {num_of_arrival} Arrival\n'
        )
     
    time_next_event_arrival = simulation_time + exponen(mean_inter_arrival_time)
    total_customer_arrived += 1

    if server_status:
        number_in_queue += 1
        assert number_in_queue <= num_of_delays_required, 'Queue is Full!!' 
        times_of_arrival.put(simulation_time)
    else:
        inc_num_customer_delayed()
        
        server_status = True
        time_next_event_departure = simulation_time + exponen(mean_service_time)
        
def depart(mean_service_time):
    global num_of_departure, time_next_event_arrival, \
        total_customer_arrived, number_in_queue, \
        area_num_in_queue, area_server_status, \
        num_customers_delayed, server_status, \
        time_next_event_departure, time_of_last_event, \
        times_of_arrival, total_delays
        
    num_of_departure += 1
    with open(OUTPUT_FILE_DIR+EVENT_ORDERS_FILE, "a+") as event:
        event.write(
            f'{num_of_event}. Next event: Customer {num_of_departure} Departure\n'
        )
        
    if number_in_queue == 0:
        server_status = False
        time_next_event_departure = math.inf
    else:
        number_in_queue -= 1
        total_delays += (simulation_time - times_of_arrival.get())
        
        inc_num_customer_delayed()
        time_next_event_departure = simulation_time + exponen(mean_service_time)
        
def generate_report():
    with open(OUTPUT_FILE_DIR+RESULT_FILE, "a+") as result:
        result.write(
            f'\nAvg delay in queue: {format(total_delays/num_customers_delayed, ".6f")} minutes\n'
            f'Avg number in queue: {format(area_num_in_queue/simulation_time, ".6f")}\n'
            f'Server utilization: {format(area_server_status/simulation_time, ".6f")}\n'
            f'Time simulation ended: {format(simulation_time, ".6f")} minutes\n'
        )
        
def main():
    # Reading inputs from `in.txt`
    with open(INPUT_FILE_DIR, "r") as inputs:
        inputs = inputs.read().split(' ')
        # print(inputs)

    mean_inter_arrival_time = float(inputs[0])
    mean_service_time = float(inputs[1])
    num_of_delays_required = int(inputs[2])
    
    if not os.path.exists(OUTPUT_FILE_DIR):
            os.makedirs(OUTPUT_FILE_DIR)
    with open(OUTPUT_FILE_DIR+RESULT_FILE, "a+") as results:
        results.write(
            f'----Single-Server Queueing System----\n\n'
            f'Mean inter-arrival time: {format(mean_inter_arrival_time, ".6f")} minutes\n'
            f'Mean service time: {format(mean_service_time, ".6f")} minutes\n'
            f'Number of customers: {num_of_delays_required}\n'
        )
    
    # Initialize the Simulation
    initialize_simulation(mean_inter_arrival_time)
    
    global num_of_event
    while (num_customers_delayed < num_of_delays_required):
        num_of_event += 1
        
        next_event_type = timing()
        update_time_avg_stats()
        
        if (next_event_type == 1):
            arrive(mean_inter_arrival_time, mean_service_time, num_of_delays_required)
        elif (next_event_type == 2):
            depart(mean_service_time)
    
    return generate_report()
            
if __name__ == "__main__":
    main()

