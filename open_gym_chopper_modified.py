import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

class OceanScape(Env):
    def __init__(self):
        super(OceanScape, self).__init__()

        # Define a 2-D observation space
        self.observation_shape = (600, 800, 3)
        self.canvas_size = self.observation_shape[:2]
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)

        # Define an action space ranging from 0 to 4
        #{0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}
        self.action_space = spaces.Discrete(5,)

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        # Define elements present inside the environment
        self.elements = []

        # Maximum battery can take at once
        self.max_battery = 1000

        # Permissible area of buoy to be
        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int (self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]


        ''' wind elements: a vector value for each point '''

        self.wind_cost_multiplier = 1
        #self.observation_shape = (600, 800, 3)
        self.wind_xgrid = np.concatenate(
            (
                np.ones((10, 51)),
                -np.ones((21, 51))
            ), axis=0
        )
        self.wind_ygrid = np.zeros(self.canvas_size)


# I need to store all the visited points to plot them
# Agent will have full access to current environment and map
# (in future -- may make reality more complicated than ideal observation)


# reset to initial state: Fixed start and goal locations, Semi random wind
# field


    def reset(self):
        # Reset the fuel consumed
        self.fuel_left = self.max_fuel

        self.buoy = Buoy() #TODO: you're stopping here

        # Reset the reward
        self.ep_return = 0 


        self.elements = [self.chopper]



# Step: movement is omni, but fuel budget is reduced based on direction

    def step(self, action):
        print(action)
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the fuel counter
        self.fuel_left -= 1

        # Reward for executing a step.
        reward = 1

        # apply the action to the chopper
        if action == 0:
            self.chopper.move(0,5)
        elif action == 1:
            self.chopper.move(0,-5)
        elif action == 2:
            self.chopper.move(5,0)
        elif action == 3:
            self.chopper.move(-5,0)
        elif action == 4:
            self.chopper.move(0,0)


        # Increment the episodic return
        if self.position in goal_region:
            self.ep_return += 100
            # + giant pot for reaching end goal
        else:
            #TODO: change this!  

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        return self.canvas, reward, done, []

# Render: renders wind field, current location, and visited locations, as
# well as start and goal locations
   def render(self, mode = "human"):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas
# This is created on opencv2.  



    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}


    def close(self):
        cv2.destroyAllWindows()



class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

        self.history = []
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

        self.history.append([(self.x, self.y)])

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)



class Chopper(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Chopper, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("chopper.png") / 255.0
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


def draw_elements_on_canvas(self):
    # Init the canvas 
    self.canvas = np.ones(self.observation_shape) * 1

    def plot_path(self, path, cl='r', flag=False):
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

    # Draw the heliopter on canvas
    for elem in self.elements:
        elem_shape = elem.icon.shape
        x,y = elem.x, elem.y
        self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

    text = 'Fuel Left: {} | Rewards: {}'.format(self.fuel_left, self.ep_return)

    # Put the info on canvas 
    self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
               0.8, (0,0,0), 1, cv2.LINE_AA)
