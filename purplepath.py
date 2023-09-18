#!/bin/python

import time
import psutil
import numpy as np
import concurrent.futures
import pathfinder
from networktables import NetworkTables

fast_cores = [4, 5, 10, 11]

def find_path(field, start, goal):
  """Find path on FRC field

  Parameters
  ----------
  field : 2D array representing the field
  start : Tuple representing the start point in centimeters
  goal : Tuple representing the goal point in centimeters
  """

  # Set CPU affinity
  psutil.Process().cpu_affinity(fast_cores)

  # Record start time
  start_time = time.time()
  # Calculate path
  path = pathfinder.astar(field, start, goal)
  # Return None if path not found
  if not path: return None
  # Smooth path
  path = pathfinder.smooth_path(path)
  # Convert path from centimeters to meters
  path = [pathfinder.cm_to_m(point) for point in path]
  # Print execution time
  print(time.time() - start_time)

  return path

if __name__ == "__main__":
  # Initalize Network Tables
  ip = "10.4.18.2"
  table_name = "DriveSubsystem"
  pose_entry = "Pose"
  
  NetworkTables.initialize(server=ip)
  table = NetworkTables.getTable(table_name)
  
  # Generate field
  field = pathfinder.generate_field(2023, 0.35)

  start = pathfinder.m_to_cm((8.00, 2.50))
  goal = pathfinder.m_to_cm((15.10, 6.75))
  
  # Start worker processes
  with concurrent.futures.ProcessPoolExecutor(max_workers=len(fast_cores)) as executor:
    while True:
      # Get robot pose via NetworkTables
      pose = tuple(table.getNumberArray(pose_entry, (8.00, 2.50)))
      pose = pathfinder.m_to_cm(pose)

      # Skip if invalid
      if field[pose] != 0: continue
      
      # Submit path for calculation, get future result asynchronously
      future = executor.submit(find_path, field, pose, goal)

      # Print result
      print(future.result())
