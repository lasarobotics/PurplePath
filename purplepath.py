#!/bin/python

import time
import psutil
import numpy as np
import multiprocessing as mp
import concurrent.futures
import pathfinder

def find_path(field, start, goal):
  """Find path on FRC field

  Parameters
  ----------
  field : 2D array representing field
  queue : Queue of tuples containing start/goal points to calculate path
  """

  # Record start time
  start_time = time.time()
  # Calculate path
  path = pathfinder.astar(field, start, goal)
  # Check if path found
  if not path: return None
  # Smooth path
  path = pathfinder.smooth_path(path)
  # Convert path from centimeters to meters
  path = [pathfinder.cm_to_m(point) for point in path]
  # Print execution time
  print(time.time() - start_time)

  return path

if __name__ == "__main__":
  field = pathfinder.generate_field(2023, 0.35)

  start = pathfinder.m_to_cm((8.00, 2.50))
  goal = pathfinder.m_to_cm((15.10, 6.75))

  fast_cores = [4, 5, 6, 7]

  # Set CPU affinity
  psutil.Process().cpu_affinity(fast_cores)
  
  # Start worker processes
  with concurrent.futures.ProcessPoolExecutor(max_workers=len(fast_cores)) as executor:
    while True:
      # Generate random point
      x = np.random.uniform(0.0, 16.50)
      y = np.random.uniform(0.0, 8.10)
      point = pathfinder.m_to_cm((x, y))
      # Skip if invalid
      if field[point] != 0: continue
      
      # Submit path for calculation, get future result asynchronously
      future = executor.submit(find_path, field, point, goal)

      # Print result
      print(future.result())
