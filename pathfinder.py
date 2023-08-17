import os
import random
import numpy as np
import heapq
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.pyplot import figure
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
  2D array representing FRC field with obstacles
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
      if i != 0 or j != 0:
        neighbor = (x + i, y + j)
        if 0 <= neighbor[0] < array.shape[0] and 0 <= neighbor[1] < array.shape[1] and array[neighbor[0]][neighbor[1]] == 0:
          neighbors.append(neighbor)
  return neighbors

def astar(array, start, goal):
  """A* path finding algorithm

  Parameters
  ----------
  array : 2D array representing field to find path on
  start : Tuple representing start position
  goal : Tuple representing goal position

  Returns
  -------
  boolean or array
    Array of tuples representing path, or False if no path could be found
  """

  # Set neighbor distance
  neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

  # List of positions that have already been considered
  close_set = set()

  # Dictionary containing all routes we've taken
  came_from = {}

  # Scores
  gscore = {start:0}
  fscore = {start:heuristic(start, goal)}

  # Create a priority queue to store the nodes to be explored
  oheap = []
  heapq.heappush(oheap, (fscore[start], start))

  while oheap:
    current = heapq.heappop(oheap)[1]

    # If the current node is the goal, return the path
    if current == goal:
      data = []
      while current in came_from:
        data.append(current)
        current = came_from[current]
      return data
    
    # Mark the current node as closed
    close_set.add(current)

    for neighbor in get_neighbors(array, current):
      tentative_g_score = gscore[current] + heuristic(current, neighbor)

      if 0 <= neighbor[0] < array.shape[0]:
        if 0 <= neighbor[1] < array.shape[1]:                
          if array[neighbor[0]][neighbor[1]] > 0:
            continue
        else:
            # array bound y walls
            continue
      else:
        # array bound x walls
        continue

      if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
        continue

      if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
        # Update the g_score and f_score of the neighbor
        gscore[neighbor] = tentative_g_score
        fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
        # Add the neighbor to the priority queue
        heapq.heappush(oheap, (fscore[neighbor], neighbor))
        # Add the neighbor to the came_from map
        came_from[neighbor] = current

  # If the goal is not reachable, return False
  return False

def dijkstra(array, start, goal):
  """Dijkstra's algorithm for finding the shortest path between two points in a graph.

  Parameters
  ----------
  array : 2D array representing the graph
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
      data = []
      while current in came_from:
        data.append(current)
        current = came_from[current]
      return data

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

def plot_route(route, start, goal):
  """Plot route graphically

  Parameters
  ----------
  route : Array of tuples representing path
  """
  # Extract x and y coordinates from route list
  x_coords = []
  y_coords = []
  for i in (range(0,len(route))):
    x = route[i][0]
    y = route[i][1]
    x_coords.append(x)
    y_coords.append(y)

  # plot map and path
  fig, ax = plt.subplots()
  cmap = mcolors.ListedColormap(['green', 'grey', 'lightgrey'])
  ax.imshow(frc_field, cmap=cmap)
  ax.scatter(start[1], start[0], marker="*", color="yellow", s=200)
  ax.scatter(goal[1], goal[0], marker="*", color="red", s=200)
  ax.plot(y_coords, x_coords, color="black")
  plt.show()


def main():
  # start point and goal
  start = m_to_cm(11.00, 0.50)
  goal = m_to_cm(15.10, 6.75)

  generate_field(2023, 0.3)

  route = dijkstra(frc_field, start, goal)
  route = route + [start]
  route = route[::-1]

  plot_route(route, start, goal)

if __name__ == "__main__":
  main()