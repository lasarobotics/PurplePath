#!/bin/python

import time
import psutil
import multiprocessing as mp
import pathfinder

field = pathfinder.generate_field(2023, 0.35)

start = pathfinder.m_to_cm((8.00, 2.50))
goal = pathfinder.m_to_cm((15.10, 6.75))

fast_cores = [4, 5, 6, 7]

def start_process(core):
  """Start pathfinding process on specific core

  Parameters
  ----------
  core : Core number to run process on
  """

  process = mp.Process(target=find_path, args=(field, queue))
  process.start()
  psutil.Process(process.pid).cpu_affinity([core])

def find_path(field, queue):
  """Find path on FRC field

  Parameters
  ----------
  field : 2D array representing field
  queue : Queue of tuples containing start/goal points to calculate path
  """

  # Infinite loop
  while True:
    # If queue is empty, skip this loop
    if queue.empty(): continue
    # Record start time
    start_time = time.time()
    # Get start and end point
    start, goal = queue.get()
    # Calculate path
    path = pathfinder.astar(field, start, goal)
    # Smooth path
    path = pathfinder.smooth_path(path)
    # Convert path from centimeters to meters
    path = [pathfinder.cm_to_m(point) for point in path]
    # Print execution time
    print(time.time() - start_time)

if __name__ == "__main__":
  # Job queue
  queue = mp.Queue()

  # Fill queue with dummy values
  for i in range(5000):
    queue.put((start, goal))

  # Start processes
  for core in fast_cores:
    start_process(core)
