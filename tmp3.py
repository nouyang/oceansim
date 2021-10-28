# NOTE: This file generates img to check coordinate systems opencv and matplotlib

import numpy as np 
import cv2 
import matplotlib.pyplot as plt


observation_shape = np.array((400, 600, 3)) #height x width, aka rows x cols
canvas_size = observation_shape[:2]


wind_image_filename = "generated_wind_plot.png"

# Initialize wind region 
def plotWindImg():
    grid_size = 200 # pixels
    wind_grid_shape = np.ceil(canvas_size/grid_size).astype(int)
    print(f'wind grid shape {wind_grid_shape}')

    # matplotlib figsize is in inches, convert to px
    my_dpi = 100
    fig, ax = plt.subplots(figsize=(canvas_size[1]/my_dpi, 
                        canvas_size[0]/my_dpi), dpi=my_dpi)
    img = plt.imread('2tmp' + wind_image_filename)
    ax.imshow(img)#, origin='upper')

    #ax.set_aspect('equal')

    #plt.axis('equal')
    plt.arrow(0, 0, 300, 0, color='blue', width=0.03, head_width=0)
    plt.arrow(0, 0, 0, 300, color='red', width=0.03, head_width=0)

    plt.text(0,20,'origin matplot')
    plt.text(300,20,'+x matplot', color='black')
    plt.text(0,300,'+y matplot')

    plt.text(300,300, f'matplot {canvas_size}')

    #plt.gca().set_axis_off()
    plt.axis("off")
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
    plt.margins(0,0)

    wind_x_grid = np.random.random_sample(wind_grid_shape) 
    wind_y_grid = np.random.random_sample(wind_grid_shape) 
    plt.quiver(wind_x_grid, wind_y_grid, color='g')

    plt.savefig('tmp' + wind_image_filename, transparent=True, pad_inches=0)

    #This draws an arrow from (x, y) to (x+dx, y+dy).


def draw_elements_on_canvas():
    # Init the canvas 
    canvas = np.ones(observation_shape) * 1

    # Dictionary containing some colors
    colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0), 'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(), 'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

    # NOTE: circle method is in x,y
    font = cv2.FONT_HERSHEY_COMPLEX
    #canvas = cv2.circle(canvas, origin, radius=10, 
    #                         color=(0, 128, 0), thickness=-1) 

    cv2.arrowedLine(canvas, (0, 0), (200, 0), colors['blue'], 3, 8, 0, 0.1)

    text = 'x = 0 cv2'
    canvas = cv2.putText(canvas, text, (0,10) , font,  0.6,
                         colors['blue'], 1, cv2.LINE_AA)
    text = '+x cv2'
    canvas = cv2.putText(canvas, text, (200,10) , font,  0.6,
                         colors['blue'], 1, cv2.LINE_AA)

    cv2.arrowedLine(canvas, (0, 0), (0, 200), colors['red'], 3, 8, 0, 0.1)
    #cv2.line(whiteblankimage, pt1=(100,300), pt2=(400,300), color=(0,0,255), thickness=10)


    text = 'y = 0 cv2'
    canvas = cv2.putText(canvas, text, (20,20) , font,  0.6,
                         colors['red'], 1, cv2.LINE_AA)
    text = '+y cv2'
    canvas = cv2.putText(canvas, text, (20,200) , font,  0.6,
                         colors['red'], 1, cv2.LINE_AA)

    text = f'OpenCV coordinate system, shape: {observation_shape}'
    canvas = cv2.putText(canvas, text, (100,100) , font,  0.6,
                         colors['black'], 1, cv2.LINE_AA)

    cv2.imshow("Game", canvas)
    cv2.imwrite('2tmp' + wind_image_filename, 255*canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#draw_elements_on_canvas()
plotWindImg()
