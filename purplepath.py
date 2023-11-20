#!/usr/bin/env python

import os
import sys
import time
import json
import signal
import ntcore
import logging
import argparse
import numpy as np
import pathfinder
from flask import Flask, request

global name
global field

name = "PurplePath"

app = Flask(name)

@app.route('/', methods=['POST'])
def get_path():
  """Server route for responding pathfinding requests
  Must send a POST request containing an array of two JSON objects representing the start and goal

  Returns:
      string: JSON string representing list of points for path
  """
  start_time = time.perf_counter()

  # Read pose, return if invalid
  start_entry = request.json[0]
  start = (0.0, 0.0)
  try:
    start = (float(start_entry['x']), float(start_entry['y']))
  except ValueError:
    log(start_time, request, "Invalid start location!")
  if np.isnan(start[0]) or np.isnan(start[1]):
    log(start_time, request, "Invalid start location!")
    return start_entry
  start = pathfinder.m_to_cm(start)

  # Return if start is outside field
  if not 0 < start[0] < field.shape[0] or not 0 < start[1] < field.shape[1]:
    log(start_time, request, "Start location is outside field!")
    return start_entry

  # Return if start is inside an object
  if field[start] == 1:
    log(start_time, request, "Start location is inside object!")
    return start_entry

  # Read goal
  goal_entry = request.json[1]
  try:
    goal = (float(goal_entry['x']), float(goal_entry['y']))
  except ValueError:
    log(start_time, request, "Invalid goal location!")
  if np.isnan(goal[0]) or np.isnan(goal[1]):
    log(start_time, request, "Invalid goal location!")
    return start_entry
  goal = pathfinder.m_to_cm(goal)

  # Return if goal is outside field
  if not 0 < goal[0] < field.shape[0] or not 0 < goal[1] < field.shape[1]:
    log(start_time, request, "Goal location is outside field!")
    return start_entry

  # Return if goal is inside an object
  if field[goal] != 0:
    log(start_time, request, "Goal location is inside an object!")
    return start_entry

  # Find path
  path = find_path(field, start, goal)

  # Return path JSON
  if not path:
    log(start_time, request, "Path NOT found!")
    return start_entry
  else:
    log(start_time, request, "Path found!")
    return path

def find_path(field, start_point, end_point):
  """Find path on FRC field

  Args:
      field (array): 2D array representing field
      start_point (tuple): tuple representing start point in centimeters
      end_point (tuple): tuple representing end point in centimeters

  Returns:
      array: Array of tuples representing the shortest path from start to goal in meters
  """

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

def log(start_time, request, message=""):
  """Log to stdout

  Args:
      start_time (int): Start time from perf_counter
      request (dict): Request from client
      message (str, optional): Additional message. Defaults to "".
  """

  current_time = time.time()
  execution_time = time.perf_counter() - start_time
  print(str(time.ctime(int(current_time))) + " " + str('{:.3f}'.format(execution_time * 1000)) + "ms\t" + message + "\t\t\t" + str(request.json))

def signal_handler(signum, frame):
  """Handle exit signal
  """

  print()
  print("Got CTRL+C, exiting...")
  sys.exit(0)

if __name__ == "__main__":
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("year", type=int, help="FRC field year to load")
  parser.add_argument("radius", type=float, help="Robot radius in meters")
  parser.add_argument("--generate-field", action="store_true", help="Generate field and exit")
  if len(sys.argv) < 2:
    parser.print_help(sys.stderr)
    sys.exit(1)
  args = parser.parse_args()

  # Set SIGINT handler
  signal.signal(signal.SIGINT, signal_handler)

  # Generate field
  field = pathfinder.generate_field(args.year, args.radius)
  if args.generate_field: sys.exit(0)

  # Set logging level
  logging.getLogger('werkzeug').setLevel(logging.ERROR)

  # Run flask app
  app.run(threaded=True)