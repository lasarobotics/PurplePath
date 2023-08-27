#!/bin/python

import heapq
import numpy as np
from scipy import interpolate

def generate_field(year, robot_radius):
  """Generate field for year

  Parameters
  ----------
  year : Field year to load
  robot_radius: Radius of robot in meters

  Returns
  -------
  array : 2D array representing FRC field with obstacles
  """
  
  import os
  
  field = np.full((1651, 811), 0)
  
  field_cache_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'fields',
    str(year) + '.npy'
  )
  if os.path.isfile(field_cache_file):
    field = np.load(field_cache_file, allow_pickle=True)
    return field
  
  field_json_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
    'fields', 
    str(year) + '.json'
  )
  with open(field_json_file) as file:
    import json
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    obstacles = json.load(file)['obstacles']
    print(obstacles)
    
    for idx in np.ndindex(field.shape):
      point = Point(idx[0] / 100, idx[1] / 100)
      for obstacle in obstacles:
        print("Checking if " + str(point) + " is within " + obstacle['name'])
        obstacle_polygon = Polygon(obstacle['vertices'])
        if obstacle_polygon.contains(point):
          print(str(point) + " is within " + obstacle['name'] + "!")
          field[idx] = 1
          break
        if obstacle_polygon.buffer(obstacle['buffer_distance'] + robot_radius).contains(point):
          print(str(point) + " is within buffer range of " + obstacle['name'] + "!")
          field[idx] = 2
          break
    np.save(field_cache_file, field, allow_pickle=True)
    return field

def m_to_cm(point):
  """Convert xy coordinate in meters to centimeters
  
  Parameters
  ----------
  point : Tuple representing coordinates

  Returns
  -------
  tuple representing point with cm units
  """

  x, y = point
  return (int(x * 100), int(y * 100))

def cm_to_m(point):
  """Convert xy coordinate in centimeters to meters
  
  Parameters
  ----------
  point : Tuple representing coordinates

  Returns
  -------
  tuple representing point with cm units
  """

  x, y = point
  return (float(x / 100), float(y / 100))

def distance(a, b):
  """Calculates the Pythagorean distance between two points
  
  Parameters
  ----------
  a : tuple representing point in centimeters
  b : tuple representing point in centimeters

  Returns
  -------
  int or float
    Estimated distance to goal
  """
  
  return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def manhattan_distance(a, b):
  """Calculates the Manhattan distance between two points
  
  Parameters
  ----------
  a : tuple representing point in centimeters
  b : tuple representing point in centimeters

  Returns
  -------
  int or float :
    Estimated distance to goal
  """

  return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(field, node, goal):
  """Returns the neighbors of a given node in the field.

  Parameters
  ----------
  field : 2D array representing the field
  node : Tuple representing the node in centimeters
  goal : Tuple representing goal in centimeters

  Returns
  -------
  array : List of tuples representing the neighbors of the node
  int : Increment used to find neighbors
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
  """Checks if the neighbor is a turn from the current point, given the previous point

  Parameters
  ----------
  current : Tuple representing the current point
  neighbor : Tuple representing the neighbor point in centimeters
  previous : Tuple representing the previous point in centimeters

  Returns
  -------
  bool : True if the neighbor is a turn, False otherwise
  """

  current_direction = (previous[0] - current[0], previous[1] - current[1])
  new_direction = (current[0] - neighbor[0], current[1] - neighbor[1])

  return current_direction != new_direction

def astar(field, start, goal):
  """A* algorithm for finding the shortest path between two points on the field with minimal turns

  Parameters
  ----------
  field : 2D array representing the field
  start : Tuple representing the start point in centimeters
  goal : Tuple representing the goal point in centimeters

  Returns
  -------
  array : Array of tuples representing the shortest path from start to goal in centimeters
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
        field[neighbor] = 3
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

  Parameters
  ----------
  path : Array of points representing path in centimeters

  Returns
  -------
  array : Array of points representing smoothed path
  """

  # Extract x and y coordinates from path
  coords = list(zip(*path))

  # Smooth the path using spline interpolation
  tck, *rest = interpolate.splprep([coords[0], coords[1]], s=250)
  x_smooth, y_smooth = interpolate.splev(np.linspace(0, 1, 100), tck)

  return list(zip(x_smooth, y_smooth))

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
  cmap = mcolors.ListedColormap(['green', 'grey', 'lightgrey', 'red'])
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
  start = m_to_cm((2.50, 0.50))
  goal = m_to_cm((15.10, 6.75))
  # start = m_to_cm((8.00, 5.00))
  # goal = m_to_cm((14.50, 1.50))

  # Calculate path
  path = astar(field, start, goal)
  path = smooth_path(path)

  # Print path
  print(path)

  # Visualize path
  plot_path(field, path, start, goal)

  # Convert path units back to meters
  path = [cm_to_m(point) for point in path]