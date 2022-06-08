import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
#import gym
import random
import math
from numpy.random import default_rng
rng = default_rng()

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

# Dictionary containing some colors
# SPECIFIC TO OPENCV (ordered as Scalar(blue_component,green_component,red_component[,alpha_component]))
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255),
        'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0),
        'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (125, 125, 125), 'rand':
        np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray': (50, 50, 50),
        'light_gray': (220, 220, 220)
        }

class OceanScape(Env):
    def __init__(self):
        super(OceanScape, self).__init__()
        self.grid_width = 50 # pixels
        self.observation_shape = np.array((300, 300, 3)) #height x width, aka rows x cols
        self.canvas_size = self.observation_shape[:2]

        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)

        self.action_space = spaces.Discrete(5,)

        self.canvas = np.ones(self.observation_shape) * 1
        self.elements = []

        self.max_battery = 1000

        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int (self.observation_shape[0] * 0.9)
        self.x_max = self.observation_shape[1]

        self.wind_grid_shape = np.ceil(self.canvas_size/self.grid_width
                ).astype(int)
        print(f'{self.canvas_size=}', f'{self.grid_width=}', f'{self.wind_grid_shape=}')

        x = np.linspace(0, self.observation_shape[0], self.wind_grid_shape[0]) 
        y = np.linspace(0, self.observation_shape[1], self.wind_grid_shape[1]) 
        self.xx, self.yy = np.meshgrid(x, y) 
        self.xx, self.yy = np.meshgrid(x, y) 
        self.wind_y_grid = np.zeros(self.wind_grid_shape) 

        # Canvas size is defined in y, x
        # Goal is defined in x, y
        self.goal = (-np.inf, -np.inf)
        self.goal_radius = 50

        self.wind_cost_multiplier = 1

        self.wind_image_filename = "generated_wind_plot.png"
        #x, y, u, v = np.random.random((4,10))
        self.plotWindImg()
        print('plotted wind img in init')

    def plotWindImg(self):
        # matplotlib figsize is in inches, convert to px
        my_dpi=100
        plt.figure(figsize=(self.canvas_size/my_dpi) )
        self.wind_x_grid = np.random.random_sample(self.wind_grid_shape) 
        self.wind_y_grid = np.random.random_sample(self.wind_grid_shape) 
        #self.wind_x_grid = rng.standard_normal(self.wind_grid_shape)
        print('*' * 30)
        #print(f'{self.wind_grid_shape[0].s)
        print(f'{self.wind_grid_shape=}, {self.grid_width=}')
        print('')
        print('!------- generated random wind ---!')
        print(f'{self.wind_x_grid.shape=}')
        print(f'{self.wind_x_grid.shape=}')
        print(f'{self.xx.shape=}')
        print('')
        #print(f'{self.xx, self.yy, 
            #self.wind_x_grid, self.wind_y_grid)
        
        plt.quiver(self.xx, self.yy, 
        self.wind_x_grid + self.grid_width/2,
            self.wind_y_grid + (self.grid_width/2), color='green') # green

        plt.grid()
        plt.savefig(self.wind_image_filename, transparent=True)

    def getWindStrength(self):
        x_pos, y_pos = self.buoy.get_position()
        column = np.floor(x_pos/self.grid_width).astype(int)
        row = np.floor(y_pos/self.grid_width).astype(int)

        try:
            x_strength = self.wind_x_grid[row][column]
            y_strength = self.wind_y_grid[row][column]
            return x_strength, y_strength
        except:
            print(f'failed! {x_pos}, {y_pos}, {column}, {row}')
            print(f'vs shape of {self.wind_x_grid}, {self.wind_x_grid.shape}')
            print(f'vs shape of {self.wind_y_grid}, {self.wind_y_grid.shape}')

    def reset(self):
        self.batt_left = self.max_battery
        self.ep_return = 0 

        # Determine a place to intialise the chopper in
        x = random.randrange(int(self.observation_shape[1] * 0.05), int(self.observation_shape[1] * 0.10))
        y = random.randrange(int(self.observation_shape[0] * 0.15), int(self.observation_shape[0] * 0.20))

        self.buoy = Buoy("buoy", self.x_max, self.x_min, self.y_max, self.y_min)
        self.buoy.set_position(x,y)

        self.elements = [self.buoy]
        self.goal = (int(np.random.rand() * self.canvas_size[1]), 
                    int(np.random.rand() * self.canvas_size[0]))

        self.wind_x_grid = np.random.random(self.wind_grid_shape) 
        self.wind_y_grid = np.random.random(self.wind_grid_shape) 
        #self.wind_x_grid = np.random.random_sample(self.wind_grid_shape) 
        #self.wind_y_grid = np.random.random_sample(self.wind_grid_shape) 
        self.plotWindImg()

        self.canvas = np.ones(self.observation_shape) * 1
        self.draw_elements_on_canvas() 

        return self.canvas 

    def inGoalRegion(self):
        b_x, b_y = self.buoy.get_position()
        dist = np.sqrt( (self.goal[0] - b_x)**2 + (self.goal[1] - b_y)**2)
        if dist <= self.goal_radius:
            #print('distance', dist)
            print(f'hurray! in goal region {self.goal},'
                  f' posit {b_x}, {b_y}, dist {dist}')
            return True
        return False


    def step(self, action):
        done = False
        assert self.action_space.contains(action), "Invalid Action"
        self.batt_left -= 1
        move_amt = 32 
        if action == 0:
            self.buoy.move(0,move_amt)
            move_x, move_y = 0, 1
        elif action == 1:
            self.buoy.move(0,-move_amt)
            move_x, move_y = 0, -1
        elif action == 2:
            self.buoy.move(move_amt,0)
            move_x, move_y = 1, 0
        elif action == 3:
            self.buoy.move(-move_amt,0)
            move_x, move_y = -1, 0
        elif action == 4:
            self.buoy.move(0,0)
            move_x, move_y = 0, 0

        wind_x, wind_y = self.getWindStrength()
        self.batt_left -= math.hypot(move_x - wind_x, move_y - wind_y)

        if self.inGoalRegion(): 
            self.ep_return += 0.1 * (self.max_battery - self.batt_left)
            print(f'reached goal, final return {self.ep_return}')
            done = True
        if self.batt_left <= 0:
            self.ep_return = -10
            print(f'oops ran out of battery, final return {self.ep_return}')
            print(f'final position {self.buoy.get_position()}')
            done = True

        # Draw elements on the canvas
        #self.draw_elements_on_canvas() # REMOVE THIS?
        reward = self.ep_return
        return self.canvas, reward, done, {}


# Render: renders wind field, current location, and visited locations, as
# well as start and goal locations
    def render(self, mode = "norender"):#"human"):
        assert mode in ["human", "norender"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            cv2.waitKey(10)
        else:
            return self.canvas

    def get_action_meanings(self):
        return {0: "Right", 1: "Left", 2: "Down", 3: "Up", 4: "Do Nothing"}

    def close(self):
        cv2.waitKey(2500)
        cv2.destroyAllWindows()

    def draw_elements_on_canvas(self):
        # Init the canvas 
        self.canvas = np.ones(self.observation_shape) * 1

        # NOTE: circle method is in x,y
        self.canvas = cv2.circle(self.canvas, self.goal, radius=self.goal_radius-32, 
                                 color=(0, 128, 0), thickness=-1) 

        # Draw the heliopter on canvas
        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            # WARNING: Note, since this is matrix notation, put y coord
            # (rows) first 
            self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = elem.icon
            
            breadcrumbs = elem.get_history()[:-1]

            for crumb in breadcrumbs:
                p_x, p_y = crumb
                # Use less efficient method for now (circle per)
                # NOTE: circle method is in x,y
                self.canvas = cv2.circle(self.canvas, (p_x+32, p_y+32), radius=2,
                                         color=(0, 0, 255), thickness=-1) 

        text = f'Batt Left: {self.batt_left:0.0f} | Rewards: {self.ep_return:.2f} ' \
        f'| Goal: {self.goal} Radius {self.goal_radius}| Loc: {x, y}'

        self.canvas = cv2.putText(self.canvas, text, (10,20), font,  0.6, (0,0,0), 1, cv2.LINE_AA)

        self.drawWindBoundaries()
        self.drawWindArrows()

        canvas = cv2.circle(self.canvas, (0,0), radius=10, 
                                 color=(0, 128, 0), thickness=-1) 

        #cv2.arrowedLine(self.canvas, (0, 0), (200, 0), colors['blue'], 3, 8, 0, 0.1)

        #canvas = cv2.putText(canvas, text, (0,10) , font,  0.6,
                              #colors['blue'], 1, cv2.LINE_AA)
        print('done drawing elements on canvas')

    def drawWindArrows(self):
        offset = np.ceil(0.2 * self.grid_width).astype(int)

        for index, x in np.ndenumerate(self.wind_x_grid):
            j, k = index
            dx = self.wind_x_grid[j][k] * int(self.grid_width / 2)
            dx = np.ceil(dx).astype(int)
            dy = self.wind_y_grid[j][k] * int(self.grid_width/2)
            dy = np.ceil(dy).astype(int)

            start_x = j * self.grid_width + offset
            start_y = k * self.grid_width + offset
            
            line_thickness=1
            tip_length=0.2 
            cv2.arrowedLine(self.canvas, 
                            (start_y, start_x), 
                            (start_y + dy, start_x + dx),
                            colors['black'], line_thickness, 8, 0, tip_length)
            
    def drawWindBoundaries(self):
        print('drwaing wind boundaries')
        height, width = self.canvas_size
        for x in range(self.wind_grid_shape[1]): 
            x_pos = x * self.grid_width
            cv2.line(self.canvas, pt1=(x_pos, 0), pt2=(x_pos, height),
                     color=colors['red'], thickness=1)
        for y in range(self.wind_grid_shape[0]): 
            y_pos = y * self.grid_width
            cv2.line(self.canvas, pt1=(0, y_pos), pt2=(width, y_pos),
                     color=colors['blue'], thickness=1)

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

        self.history.append((self.x, self.y))

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def get_history(self):
        return self.history 


class Buoy(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Buoy, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("argo_buoy.png") / 255.0
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

# NOTES: 
    # You can use cv2.circle() function opencv module:
    # image = cv.circle(image, centerOfCircle, radius, color, thickness)
    # Keep radius as 0 for plotting a single point and thickness as a negative number for filled circle
            #text = 'Rewards: {}'.format(self.ep_return)
            #self.canvas = cv2.putText(self.canvas, text, (10,50), font,  
            #           2, (255,0,0), 1, cv2.LINE_AA)

#plt.imshow(obs)
#plt.show()

#screen = env.render(mode = "rgb_array")
#plt.imshow(screen)
#plt.show()

def main():
    env = OceanScape()
    obs = env.reset()

    while True:
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        env.render(mode='human')
        plt.show()

        if done == True:
            cv2.waitKey(0)
            break

if __name__ == '__main__':
    main()