#!/bin/python

import os
import random
import numpy as np
import heapq
import json
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import figure
from scipy.special import comb
from scipy import interpolate
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

frc_field = np.full((1650, 810), 0)

def generate_field(year, robot_radius):
  """Generate field for year

  Parameters
  ----------
  year : Field year to load

  Returns
  -------
  array : 2D array representing FRC field with obstacles
  """

  global frc_field
  if os.path.isfile(str(year) + '.npy'):
    frc_field = np.load(str(year) + '.npy', allow_pickle=True)
    return
  
  with open('fields/' + str(year) + '.json') as file:
    obstacles = json.load(file)['obstacles']
    print(obstacles)
    
    for idx in np.ndindex(frc_field.shape):
      point = Point(idx[0] / 100, idx[1] / 100)
      for obstacle in obstacles:
        print("Checking if " + str(point) + " is within " + obstacle['name'])
        obstacle_polygon = Polygon(obstacle['vertices'])
        if obstacle_polygon.contains(point):
          print(str(point) + " is within " + obstacle['name'] + "!")
          frc_field[idx] = 1
          break
        if obstacle_polygon.buffer(obstacle['buffer_distance'] + robot_radius).contains(point):
          print(str(point) + " is within buffer range of " + obstacle['name'] + "!")
          frc_field[idx] = 2
          break
    np.save(str(year), frc_field, allow_pickle=True)
        
def m_to_cm(x, y):
  """Convert xy coordinate in meters to centimeters
  
  Parameters
  ----------
  x : x value
  y : y value

  Returns
  -------
  tuple representing point with cm units
  """
  return int(x * 100), int(y * 100)

def heuristic(a, b):
  """Heuristic function for path scoring 
  
  Parameters
  ----------
  a : tuple representing point
  b : tuple representing point

  Returns
  -------
  int or float
    Estimated distance to goal
  """
  return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def get_neighbors(array, node):
  """Returns the neighbors of a given node in the graph.

  Parameters
  ----------
  array : 2D array representing the graph
  node : Tuple representing the node

  Returns
  -------
  array : List of tuples representing the neighbors of the node
  """

  neighbors = []
  x, y = node
  for i in (-1, 0, 1):
    for j in (-1, 0, 1):
      if i == 0 and j == 0: continue
      neighbor = (x + i, y + j)
      if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1] and array[neighbor[0]][neighbor[1]] == 0:
        neighbors.append(neighbor)
  return neighbors

  # If the goal is not reachable, return False
  return False

def is_turn(current, neighbor, previous):
  """Checks if the neighbor is a turn from the current point, given the previous point

  Parameters
  ----------
  current : Tuple representing the current point
  neighbor : Tuple representing the neighbor point
  previous : Tuple representing the previous point

  Returns
  -------
  bool : True if the neighbor is a turn, False otherwise
  """

  current_direction = tuple(x - y for x, y in zip(previous, current))
  new_direction = tuple(x - y for x, y in zip(current, neighbor))

  return current_direction != new_direction

def astar(array, start, goal):
  """A* algorithm for finding the shortest path between two points in a graph.

  Parameters
  ----------
  array : 2D array representing the field
  start : Tuple representing the start node
  goal : Tuple representing the goal node

  Returns
  -------
  array : Array of tuples representing the shortest path from start to goal
  """

  # List of positions that have already been considered
  close_set = set()

  # Dictionary containing all routes we've taken
  came_from = {}

  # Scores
  g_score = {start: 0}
  f_score = {start: heuristic(start, goal)}

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
        current = came_from[current]
      return path[::-1]

    # Mark the current node as closed
    close_set.add(current)

    # For each neighbor of the current node
    for neighbor in get_neighbors(array, current):
      # If the neighbor is not closed and the tentative g_score is less than the current g_score
      if neighbor not in close_set and g_score[current] + array[neighbor[0]][neighbor[1]] < g_score.get(neighbor, float('inf')):
        # Update the g_score and f_score of the neighbor
        g_score[neighbor] = g_score[current] + array[neighbor[0]][neighbor[1]]
        f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
        # Add the neighbor to the priority queue
        heapq.heappush(oheap, (f_score[neighbor], neighbor))
        # Add the neighbor to the came_from map
        came_from[neighbor] = current

  # If the goal is not reachable, return False
  return False

def astar_with_turns(array, start, goal):
  """A* algorithm for finding the shortest path between two points in a graph, with the fewest number of turns.

  Parameters
  ----------
  array : 2D array representing the field
  start : Tuple representing the start node
  goal : Tuple representing the goal node

  Returns
  -------
  array : Array of tuples representing the shortest path from start to goal
  """

  # Initialize the data structures
  close_set = set()
  came_from = {start: start}
  g_score = {start: 0}
  turn_score = {start: 0}
  turn_penalty = 5
  f_score = {start: g_score[start] + heuristic(start, goal)}

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

    # For each neighbor of the current node
    for neighbor in get_neighbors(array, current):
      # If the neighbor is not closed and the tentative g_score is less than the current g_score
      if neighbor not in close_set and g_score[current] + array[neighbor[0]][neighbor[1]] < g_score.get(neighbor, float('inf')):
        # Add the neighbor to the came_from map
        came_from[neighbor] = current
        # Update the g_score and turn_score of the neighbor
        g_score[neighbor] = g_score[current] + array[neighbor[0]][neighbor[1]]
        turn_score[neighbor] = turn_score[current] + 1
        f_score[neighbor] = g_score[neighbor] + turn_score[neighbor] + heuristic(neighbor, goal)
        if is_turn(current, neighbor, came_from[current]):
          f_score[neighbor] -= turn_penalty
        # Add the neighbor to the priority queue
        heapq.heappush(oheap, (f_score[neighbor], neighbor))

  # If the goal is not reachable, return False
  return False

def smooth_path(path):
  """Smooth path

  Parameters
  ----------
  path : Array of points representing path

  Returns
  -------
  array : Array of points representing smoothed path
  """

  # Extract x and y coordinates from path
  coords = list(zip(*path))

  # Smooth the path using spline interpolation
  tck, *rest = interpolate.splprep([coords[0], coords[1]])
  x_smooth, y_smooth = interpolate.splev(np.linspace(0, 1, 80), tck)

  return list(zip(x_smooth, y_smooth))

def plot_path(path, start, goal):
  """Plot path graphically

  Parameters
  ----------
  path : Array of tuples representing path
  """
  
  # Extract x and y coordinates from path
  coords = list(zip(*path))

  # plot map and path
  fig, ax = plt.subplots()
  cmap = mcolors.ListedColormap(['green', 'grey', 'lightgrey'])
  ax.imshow(frc_field, cmap=cmap)
  ax.scatter(start[1], start[0], marker="*", color="yellow", s=200)
  ax.scatter(goal[1], goal[0], marker="*", color="purple", s=200)
  ax.plot(coords[1], coords[0], color="blue")
  plt.show()


def main():
  # start point and goal
  start = m_to_cm(11.00, 0.50)
  goal = m_to_cm(15.10, 6.75)

  # Generate FRC field
  generate_field(2023, 0.35)

  # Calculate and smooth path
  path = astar_with_turns(frc_field, start, goal)
  path = smooth_path(path)

  # Visualize path
  plot_path(path, start, goal)

if __name__ == "__main__":
  main()