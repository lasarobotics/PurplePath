import sys
import time
import ntcore
import time
import pathfinder
import logging as log
import pandas as pd
import json
import numpy as np
import wpimath.geometry
from io import StringIO
from wpiutil import wpistruct
from wpimath.geometry import Translation2d
import dataclasses

global name
global field

field = np.array(pathfinder.generate_field(2024, 0.45))

print(field)

start_time = time.perf_counter()

@wpistruct.make_wpistruct
@dataclasses.dataclass
class point:
    value: Translation2d

    

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

    print(f"path: {path}")

    return path

def log(start_time, message=""):
    """Log to stdout

    Args:
        start_time (int): Start time from perf_counter
        request (dict): Request from client
        message (str, optional): Additional message. Defaults to "".
    """

    current_time = time.time()
    execution_time = time.perf_counter() - start_time
    print(str(time.ctime(int(current_time))) + "\t" + str('{:.3f}'.format(execution_time * 1000)) + "ms \t" + message + "\t\t\t\t\t")

def signal_handler(signum, frame):
    """Handle exit signal
    """

    print()
    print("Got CTRL+C, exiting...")
    sys.exit(0)

def path_creater(pathSet, x1, y1, x2, y2):
    print(f"X1: {x1} Y1: {y1} X2: {x2} Y2: {y2}")

    start_entry = tuple([x1, y1])
    start = (0.0, 0.0)

    try:
        start = (start_entry[0], start_entry[1])
    except ValueError:
        log(start_time, "Invalid start location!")
        return


    if (start[0] == None) or (start[1] == None):
        log(start_time, "Invalid start location!")
        return


    start = pathfinder.m_to_cm(start)

    # Return if start is outside field
    if not 0 < start[0] < field.shape[0] or not 0 < start[1] < field.shape[1]:
        log(start_time, "Start location is outside field!")
        return

    # Return if start is inside an object
    if field[start] == 1:
        log(start_time, "Start location is inside object!")
        return

        
    # Read goal
    goal_entry = tuple([x2, y2])
    try:
        goal = (float(goal_entry[0]), float(goal_entry[1]))
    except ValueError:
        log(start_time, "Invalid goal location!")
        return


    if goal[0] == None or goal[1] == None:
        log(start_time, "Invalid goal location!")
        return


    goal = pathfinder.m_to_cm(goal)

    # Return if goal is outside field
    if not 0 < goal[0] < field.shape[0] or not 0 < goal[1] < field.shape[1]:
        log(start_time, "Goal location is outside field!")
        return


    # Return if goal is inside an object
    if field[goal] != 0:
        log(start_time, "Goal location is inside an object!")
        return


    # Find path
    path = find_path(field, start, goal)

    point_list = []

    if(path != None):
        for idx, path_pointer in enumerate(path):
            point_list.append(Translation2d(path_pointer[0], path_pointer[1]))
    
    print(f"{x2} + {y2}")
    if(len(point_list) != 0):
        print(point_list[len(point_list)-1])


    if not path:
        log(start_time, "Path NOT found!")
        return

    else:
        log(start_time, "Path found!")
        pathSet.set(point_list)


if __name__ == "__main__":
    inst = ntcore.NetworkTableInstance.getDefault()
    table = inst.getTable("SmartDashboard")

    defaultVal = [Translation2d(0, 0), Translation2d(0, 0)]

    curPosSub = table.getStructTopic("Start Point", Translation2d).subscribe(defaultVal[1])
    endPosSub = table.getStructTopic("End Point", Translation2d).subscribe(defaultVal[0])

    curPosPub = table.getStructTopic("Start Point", Translation2d).publish()
    endPosPub = table.getStructTopic("End Point", Translation2d).publish()

    pathSet = table.getStructArrayTopic("Path", Translation2d).publish()
    
    inst.startClient4("example client")
    inst.setServer("localhost") # where TEAM=190, 294, etc, or use inst.setServer("hostname") or similar
    inst.startDSClient() # recommended if running on DS computer; this gets the robot IP from the DS

    start_position = curPosSub.get()
    end_position = endPosSub.get()


    start_pos_x = round(start_position[0], 2)
    start_pos_y = round(start_position[1], 2)

    end_pos_x = round(end_position[0], 2)
    end_pos_y = round(end_position[1], 2)

    x1 = start_pos_x
    y1 = start_pos_y
    x2 = end_pos_x
    y2 = end_pos_y

    count = 0
    while True:
        start_val = curPosSub.get()
        end_val = endPosSub.get()
        
        start_val_x = round(start_val[0], 2)
        start_val_y = round(start_val[1], 2)

        if(start_val_x != x1 and start_val_y != y1):
            x1 = start_val[0]
            y1 = start_val[1]

            x2 = end_val[0]
            y2 = end_val[1]
            path_creater(pathSet, x1, y1, x2, y2)

        print(f"count: {count}")
        count+=1

        time.sleep(1)