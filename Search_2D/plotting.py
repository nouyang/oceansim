"""
Plot tools 2D
@author: huiming zhou
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

from Search_2D import env


class Plotting:
    def __init__(self, xI, xG):
        self.xI, self.xG = xI, xG
        self.env = env.Env()
        self.obs = self.env.obs_map()

    def update_obs(self, obs):
        self.obs = obs
    def plot_wind(self, xgrid, ygrid):
        # actually just dots...
        #sns.set_theme(context='talk', style='whitegrid')

# GRID
        x = np.arange(0,xgrid.shape[1]-1,1)
        y = np.arange(0,xgrid.shape[0]-1,1)
        XX, YY = np.meshgrid(x,y)

        #sns.scatterplot(XX, YY)
        #plt.plot(XX, YY, color='gray', markersize=1)
        #plt.scatter(XX, YY, color='gray', size=1)
        plt.plot(XX.flat, YY.flat, ".", color='gray', markersize='2')

# ARROWS
        # put in some arrows
        x = np.arange(0,xgrid.shape[1],10)
        y = np.arange(0,xgrid.shape[0]-10,10)
        XX, YY = np.meshgrid(x,y)
        XX = XX + 5
        YY = YY + 5
        plt.plot(XX.flat, YY.flat, ">", color='blue', markersize='15')

    '''
        XX, YY = np.meshgrid(x,y)
        print(xgrid.shape, ygrid.shape)
        print(xgrid, ygrid)
        #dim_x, dim_y = xgrid.shape
        #plt.quiver(X,Y, xgrid, ygrid, color='g')
        plt.plot(xgrid, ygrid)
        plt.plot(50,50)

        plt.quiver(xgrid, ygrid, color='gray')
        #plt.rcParams['figure.figsize'] = (10, 10)
        '''

    def plot_grid(self, name):
        # obstacles
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        # mark goal
        plt.plot(self.xI[0], self.xI[1], "b", marker='o', markersize=20)
        plt.plot(self.xG[0], self.xG[1], "g", marker='X', markersize=25)

        xy_goal = (self.xG[0], self.xG[1])
        xy_start = (self.xI[0], self.xI[1])

        plt.annotate('START',
             xy=xy_start, 
             xytext=(15, -20),
             textcoords='offset points')

        plt.annotate('GOAL',
             xy=xy_goal, 
             xytext=(-15, 25),
             textcoords='offset points')
        # xy is the point (x, y) to annotate
        # xytext is the position (x, y) to place the text at (default: xy). The coordinate system is determined by textcoords.
        # textcoords is the coordinate system that xy is given in ('offset points' means the offset in points from the xy value)

        # left and right line
        plt.vlines([0,50], color='k', ymin=0, ymax=30, linewidth=8)
        # right line
        plt.hlines([0,30], color='k', xmin=0, xmax=50, linewidth=8)
        plt.vlines([20,40], color='k', ymin=0, ymax=15, linewidth=8)
        plt.plot([30,30], [15,30], color='k', linewidth=8) # top vertical


        plt.title(name)
        plt.axis("equal") # NOTE: THIS ATE AN HOUR OF MY LIFE
        #plt.rcParams["figure.figsize"]= (20,10)

    def plot_visited(self, visited, cl='gray'):
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40
            # length = 15

            if count % length == 0:
                plt.pause(0.001)
        plt.pause(1)

    def plot_path(self, path, cl='r', flag=False):
        print('plot path')
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]

        if not flag:
            plt.plot(path_x, path_y, linewidth=10, color='r',
                marker='s', markersize=2)
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)


        #plt.plot(self.xI[0], self.xI[1], color="blue", marker='o')#, markersize=15) #  s = square
        #plt.plot(self.xG[0], self.xG[1], color="green", marker='o')#, markersize=15)

        plt.pause(0.01)

    def plot_visited_bi(self, v_fore, v_back):
        if self.xI in v_fore:
            v_fore.remove(self.xI)

        if self.xG in v_back:
            v_back.remove(self.xG)

        len_fore, len_back = len(v_fore), len(v_back)

        for k in range(max(len_fore, len_back)):
            if k < len_fore:
                plt.plot(v_fore[k][0], v_fore[k][1], linewidth='3', color='gray', marker='o')
            if k < len_back:
                plt.plot(v_back[k][0], v_back[k][1], linewidth='3', color='cornflowerblue', marker='o')

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 10 == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    @staticmethod
    def color_list():
        cl_v = ['silver',
                'wheat',
                'lightskyblue',
                'royalblue',
                'slategray']
        cl_p = ['gray',
                'orange',
                'deepskyblue',
                'red',
                'm']
        return cl_v, cl_p

    @staticmethod
    def color_list_2():
        cl = ['silver',
              'steelblue',
              'dimgray',
              'cornflowerblue',
              'dodgerblue',
              'royalblue',
              'plum',
              'mediumslateblue',
              'mediumpurple',
              'blueviolet',
              ]
        return cl


'''
    def animation(self, path, visited, name):
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        plt.show()

    def animation_lrta(self, path, visited, name):
        self.plot_grid(name)
        cl = self.color_list_2()
        path_combine = []

        for k in range(len(path)):
            self.plot_visited(visited[k], cl[k])
            plt.pause(0.2)
            self.plot_path(path[k])
            path_combine += path[k]
            plt.pause(0.2)
        if self.xI in path_combine:
            path_combine.remove(self.xI)
        self.plot_path(path_combine)
        plt.show()

    def animation_ara_star(self, path, visited, name):
        self.plot_grid(name)
        cl_v, cl_p = self.color_list()

        for k in range(len(path)):
            self.plot_visited(visited[k], cl_v[k])
            self.plot_path(path[k], cl_p[k], True)
            plt.pause(0.5)

        plt.show()

    def animation_bi_astar(self, path, v_fore, v_back, name):
        self.plot_grid(name)
        self.plot_visited_bi(v_fore, v_back)
        self.plot_path(path)
        plt.show()
'''