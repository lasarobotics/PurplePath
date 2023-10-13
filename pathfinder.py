#!/bin/python

import json
import heapq
import numpy as np
from scipy import interpolate

def generate_field(year, robot_radius):
  """Generate field for year

  Args:
      year (int): Field year to load
      robot_radius (float): Radius of robot in meters

  Returns:
      array: 2D array representing field
  """
  
  import os

  FIELD_LENGTH = 16.50
  FIELD_WIDTH = 8.10
  WALL_BUFFER = 0.05
  
  # 2D array representing field
  field = np.full((int(FIELD_LENGTH * 100) + 1, int(FIELD_WIDTH * 100) + 1), 0)
  
  # Field cache file
  field_cache_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'fields',
    str(year) + '.npy'
  )
  
  # If field cache file exists, read it and return fieldbhgfdxfgfdravbbv
  if os.path.isfile(field_cache_file):
    field = np.load(field_cache_file, allow_pickle=True)
    return field
  
  # Field json file
  field_json_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    'fields', 
    str(year) + '.json'
  )
  with open(field_json_file) as file:
    import json
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    
    # Read obstacles from file
    obstacles = json.load(file)['obstacles']
    obstacles = [
      (obstacle['name'], obstacle['buffer_distance'], Polygon(obstacle['vertices']))
      for obstacle in obstacles
    ]
    print(obstacles)
    
    # Iterate over every square cm
    for idx in np.ndindex(field.shape):
      point = Point(idx[0] / 100, idx[1] / 100)
      # Iterate over each obstacle
      for name, buffer_distance, shape in obstacles:
        # Check if point is within obstacle
        print("Checking if " + str(point) + " is within " + name)
        if shape.contains(point):
          print(str(point) + " is within " + name + "!")
          field[idx] = 1
          break
        # Check if point is within buffer range of obstacle
        if shape.buffer(buffer_distance + robot_radius).contains(point):
          print(str(point) + " is within buffer range of " + name + "!")
          field[idx] = 2
          break
      # If point has already been identified as obstacle or buffer zone, continue
      if field[idx] != 0: continue
      # Check if point is close to field walls
      if point.x <= robot_radius + WALL_BUFFER or point.x >= FIELD_LENGTH - (robot_radius + WALL_BUFFER) \
        or point.y <= robot_radius + WALL_BUFFER or point.y >= FIELD_WIDTH - (robot_radius + WALL_BUFFER):
        print(str(point) + " is within buffer range of field walls!")
        field[idx] = 2

    # Save field into cache file and return
    np.save(field_cache_file, field, allow_pickle=True)
    return field

def m_to_cm(point):
  """Convert xy coordinate in meters to centimeters

  Args:
      point (tuple): Tuple representing coordinate

  Returns:
      tuple: tuple representing point in centimeters
  """

  x, y = point
  return (int(x * 100), int(y * 100))

def cm_to_m(point):
  """Convert xy coordinate in centimeters to meters

  Args:
      point (tuple): Tuple representing coordinate

  Returns:
      tuple: tuple representing point in meters
  """

  x, y = point
  return (float(x / 100), float(y / 100))

def distance(a, b):
  """Calculates the Pythagorean distance between two points

  Args:
      a (tuple): tuple representing point in centimeters
      b (tuple): tuple representing point in centimeters

  Returns:
      int or float: Estimated distance to goal
  """
  
  return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def manhattan_distance(a, b):
  """Manhattan distance

  Args:
      a (tuple): tuple representing point in centimeters
      b (tuple): tuple representing point in centimeters

  Returns:
      int or float: Estimated distance to goal
  """

  return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(field, node, goal):
  """Returns the neighbors of a given node in the field.

  Args:
      field (tuple): 2D array representing the field
      node (tuple): Tuple representing the node in centimeters
      goal (tuple): Tuple representing goal in centimeters

  Returns:
      array: List of tuples representing the neighbors of the node
      int: Increment used to find neighbors
  """
  
  increment = 1 if manhattan_distance(node, goal) < 35 else 25
  neighbors = []
  x, y = node
  for i in (-increment, 0, +increment):
    for j in (-increment, 0, +increment):
      if i == 0 and j == 0: continue
      neighbor = (x + i, y + j)
      if 0 <= neighbor[0] < field.shape[0] and 0 <= neighbor[1] < field.shape[1]:
        neighbors.append(neighbor)
  return neighbors, increment

def is_turn(current, neighbor, previous):
  """Checks if going to the neighbor from the current point requires a turn, given the previous point

  Args:
      current (tuple): Tuple representing current point in centimeters
      neighbor (tuple): Tuple representing neighbor point in centimeters
      previous (tuple): Tuple representing previous point in centimeters

  Returns:
      bool: True if neighbor is a turn, False otherwise
  """

  current_direction = (previous[0] - current[0], previous[1] - current[1])
  new_direction = (current[0] - neighbor[0], current[1] - neighbor[1])

  return current_direction != new_direction

def astar(field, start, goal):
  """A* algorithm for finding the shortest path between two points on the field with minimal turns

  Args:
      field (array): 2D array representing field
      start (tuple): Tuple representing start point in centimeters
      goal (tuple): Tuple representing goal point in centimeters

  Returns:
      array: Array of tuples representing the shortest path from start to goal in centimeters
  """
  
  # Distance weights to neighbor nodes
  neighbor_distances = { 0: 1.4, 1: 1.0, 2: 1.4, 3: 1.0, 4: 1.0, 5: 1.4, 6: 1.0, 7: 1.4 }

  ## List of positions that have already been considered
  close_set = set()
  
  # Dictionary containing all routes we've taken
  came_from = { start: start }
  
  # Scores
  turn_penalty = 10
  g_score = { start: 0 }
  f_score = { start: distance(start, goal) }

  # Create a priority queue to store the nodes to be explored
  oheap = []
  heapq.heappush(oheap, (f_score[start], start))

  # While there are nodes to be explored
  while oheap:
    # Get the node with the lowest f_score
    current = heapq.heappop(oheap)[1]

    # If the current node is the goal, return the path
    if current == goal:
      path = []
      while current in came_from:
        path.append(current)
        if current == start: break
        current = came_from[current]
      return path[::-1]

    # Mark the current node as closed
    close_set.add(current)
    
    # Get neighbors
    neighbors, increment = get_neighbors(field, current, goal)

    # Scored neighbor heap
    scored_neighbors = []

    # For each neighbor of the current node
    for idx, neighbor in enumerate(neighbors):
      # If neighbor is not clear, skip
      if field[neighbor] != 0:
        continue
      # If neighbor is estimated to be a worse path, skip
      neighbor_goal_distance = distance(neighbor, goal)
      if neighbor_goal_distance > f_score[current]:
        continue
      # If the neighbor is not closed and the current f_score is greater than the neighbor f_score
      if neighbor not in close_set and f_score[current] > f_score.get(neighbor, 0):
        # Add the neighbor to the came_from map
        came_from[neighbor] = current
        # Update the g_score and f_score of the neighbor
        g_score[neighbor] = g_score[current] + neighbor_distances.get(idx) * increment
        f_score[neighbor] = g_score[neighbor] + neighbor_goal_distance
        # Penalize turns
        if is_turn(current, neighbor, came_from[current]):
          f_score[neighbor] += turn_penalty
        # Add the neighbor to list of scored neighbors
        heapq.heappush(scored_neighbors, (f_score[neighbor], neighbor))
    
    # Add best 2 neighbors to priority queue
    for i in range(min(len(scored_neighbors), 2)):
      heapq.heappush(oheap, heapq.heappop(scored_neighbors))

  # If the goal is not reachable, return False
  return False

def smooth_path(path):
  """Smooth path

  Args:
      path (array): Array of tuples representing path in centimeters

  Returns:
      array: Array of points in smoothed path
  """

  # Extract x and y coordinates from path
  coords = list(zip(*path))

  # Smooth the path using spline interpolation
  tck, *rest = interpolate.splprep([coords[0], coords[1]], s=500)
  x_smooth, y_smooth = interpolate.splev(np.linspace(0, 1, 100), tck)

  return list(zip(x_smooth, y_smooth))

def path_to_json(path):
  """Convert path to JSON formatted string

  Args:
      path (array): List of tuples representing points

  Returns:
      str: JSON formatted string
  """
  path = [
    { "x": point[0], "y": point[1] }
    for point in path
  ]
  return json.dumps(path)

def plot_path(field, path, start, goal):
  """Plot path graphically

  Parameters
  ----------
  field : 2D array representing field
  path : Array of tuples representing path in centimeters
  """

  import matplotlib.pyplot as plt
  import matplotlib.colors as mcolors

  # Extract x and y coordinates from path
  coords = list(zip(*path))
  
  # Plot field and path
  fig, ax = plt.subplots()
  cmap = mcolors.ListedColormap(['green', 'grey', 'lightgrey'])
  ax.imshow(field, cmap=cmap)
  ax.scatter(start[1], start[0], marker='*', color='yellow', s=200)
  ax.scatter(goal[1], goal[0], marker='*', color='purple', s=200)
  ax.plot(coords[1], coords[0], color='black')
  plt.show()


if __name__ == "__main__":
  # Example of how to use PurplePath

  # Generate FRC field
  field = generate_field(2023, 0.35)
  
  # Start point and goal
  start = m_to_cm((7.50, 3.50))
  goal = m_to_cm((15.10, 6.75))
  # start = m_to_cm((8.00, 5.00))
  # goal = m_to_cm((14.50, 2.50))

  # Calculate path
  path = astar(field, start, goal)
  path = smooth_path(path)

  # Visualize path
  #plot_path(field, path, start, goal)

  # Convert path units back to meters
  path = [cm_to_m(point) for point in path]
  
  # Convert path to JSON string
  path_json = path_to_json(path)
  
  # Print JSON string
  print(path_json)