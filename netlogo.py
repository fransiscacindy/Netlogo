import pickle
import os
import traceback
from typing import List

import cProfile
import pstats

from lib.generator.warehouse_generator import *
from pip._internal import main as pipmain
from lib.file import *
from world.warehouse import Warehouse

def setup():
    try:
        # Initialize the simulation warehouse
        assignment_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        if os.path.exists(assignment_path):
            os.remove(assignment_path)
        warehouse = Warehouse()
        
        # Populate the warehouse with objects and connections
        draw_layout(warehouse)
        # print(warehouse.intersection_manager.intersections[0].intersection_coordinate)

        # Generate initial results
        next_result = warehouse.generateResult()
        
        warehouse.initWarehouse();

        # Save the warehouse state for future ticks
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(warehouse, config_dictionary_file)

        return next_result

    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def tick():
    try:
        # print("========tick========")

        # Load the simulation state
        with open('netlogo.state', 'rb') as file:
            warehouse: Warehouse = pickle.load(file)

        print("before tick", warehouse._tick)

        # Update each object with the current warehouse context

        # Perform a simulation tick
        warehouse.tick()

        # Generate results after the tick
        next_result = warehouse.generateResult()
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(warehouse, config_dictionary_file)
        return [next_result, warehouse.total_energy, len(warehouse.job_queue), warehouse.stop_and_go,
                warehouse.total_turning]
    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def setup_py():
    def install_package(package_name):
        """Install a Python package using pip."""
        pipmain(['install', package_name])

    # List of packages to install
    packages = ["networkx", "matplotlib"]

    # Install each package
    for package in packages:
        install_package(package)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling

    result = setup()
    for _ in range(500):
        result = tick() 

    profiler.disable()  # Stop profiling

    # Print the profiling results, sorted by time taken
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumtime").print_stats(10)  # Top 10 functions

    # with open('result.txt', 'w') as result_file:
    #     result_file.write(str(result))