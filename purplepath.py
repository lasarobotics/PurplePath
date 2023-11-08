#!/bin/python

import os
import sys
import time
import json
import psutil
import ntcore
import argparse
import numpy as np
import concurrent.futures
import pathfinder

name = "PurplePath"
fast_cores = [15, 31]

def find_path(field, start_point, end_point):
  """Find path on FRC field

  Args:
      field (array): 2D array representing field
      start_point (tuple): tuple representing start point in centimeters
      end_point (tuple): tuple representing end point in centimeters

  Returns:
      array: Array of tuples representing the shortest path from start to goal in meters
  """

  # Set CPU affinity
  psutil.Process().cpu_affinity(fast_cores)
  # Calculate path
  path = pathfinder.astar(field, start_point, end_point)
  # Return None if path not found
  if not path: return None
  # Smooth path
  path = pathfinder.smooth_path(path)
  # Convert path from centimeters to meters
  path = [pathfinder.cm_to_m(point) for point in path]

  path_json = pathfinder.path_to_json(path)

  return path_json

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("year", type=int, help="FRC field year to load")
  parser.add_argument("radius", type=float, help="Robot radius in meters")
  parser.add_argument("--test", action="store_true", help="Run in test mode")
  if len(sys.argv) < 2:
    parser.print_help(sys.stderr)
    sys.exit(1)
  args = parser.parse_args()

  # Initalize Network Tables
  inst = ntcore.NetworkTableInstance.getDefault()
  table = inst.getTable(name)
  pose_subscriber = table.getStringTopic("Pose").subscribe("")
  goal_subscriber = table.getStringTopic("Goal").subscribe("")
  trajectory_publishers = []
  for idx in range(10):
    trajectory_publisher = table.getStringTopic("Trajectory" + str(idx)).publish(ntcore.PubSubOptions(periodic=0.02, keepDuplicates=True, pollStorage=10))
    trajectory_publishers.append(trajectory_publisher)

  inst.startClient4(name)
  if args.test: inst.setServer("localhost", ntcore.NetworkTableInstance.kDefaultPort4)
  else: inst.setServerTeam(418)

  # Generate field
  field = pathfinder.generate_field(args.year, args.radius)

  # Start worker processes
  with concurrent.futures.ProcessPoolExecutor(max_workers=len(fast_cores)) as executor:
    while True:
      # Get robot pose via NetworkTables
      pose_entry = pose_subscriber.get()
      goal_entry = goal_subscriber.get()

      # Continue if pose or goal invalid
      if not pose_entry or not goal_entry: continue

      # Read pose
      pose_entry = json.loads(pose_entry)
      pose = (0.0, 0.0)
      try:
        pose = (float(pose_entry['x']), float(pose_entry['y']))
      except ValueError:
        print("Not valid pose")
      if np.isnan(pose[0]) or np.isnan(pose[1]): continue
      pose = pathfinder.m_to_cm(pose)

      # Continue if pose is outside field
      if not 0 < pose[0] < field.shape[0] and 0 < pose[1] < field.shape[1]: continue

      # Read goal
      goal_entries = json.loads(goal_entry)
      if len(goal_entries) > 10: continue
      for idx, goal_entry in enumerate(goal_entries):
        goal = (goal_entry['x'], goal_entry['y'])
        goal = pathfinder.m_to_cm(goal)

        # Skip if pose or goal is obstacle
        if field[pose] != 0 or field[goal] != 0:
          trajectory_publishers[idx].set("")
          continue

        # Submit path for calculation, get future result asynchronously
        future = executor.submit(find_path, field, pose, goal)

        # Publish result to network tables
        if future.result(): trajectory_publishers[idx].set(future.result())
        else: trajectory_publishers[idx].set("")
