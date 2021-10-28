import cv2
import numpy as np
'''
img = cv2.imread('argo_buoy.png')
cv2.imshow('output', img)
cv2.waitKey(0)
'''
import matplotlib.pyplot as plt

	my_dpi=100

	fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)

	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

# 51 x 31
xgrid = np.concatenate(
    (
    np.ones((10, 51)),
    -np.ones((21, 51))
    ), axis=0
 )
ygrid = np.zeros((31,51))

plt.quiver(xgrid, ygrid, color='g')
#plt.rcParams['figure.figsize'] = (10, 10)
plt.savefig('test.png')

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

observation_shape = (400, 600, 3)
canvas = np.ones(observation_shape) * 1

canvas_size = observation_shape[:2]

goal = (int(np.random.rand() * canvas_size[1]),  #x, y
            int(np.random.rand() * canvas_size[0]))

goal_radius = 100
font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

#text = 'Batt Left: {} | Rewards: {}'.format(batt_left, ep_return)
batt_left = 100
ep_return = -1
text = f'Batt Left: {batt_left} | Rewards: {ep_return} | Goal: {goal} Radius {goal_radius}'

# Put the info on canvas 
# canvas = cv2.putText(canvas, text, (10,20), font,  0.8, (0,0,0), 1, cv2.LINE_AA)

canvas = cv2.circle(canvas, (10,100), radius=10,
                         color=(0, 128, 0), thickness=-1) 

#overlay = cv2.imread('wind_field_overlay_trans.png')#, cv2.IMREAD_UNCHANGED)
overlay = cv2.imread('wind_field_overlay_trans.png', cv2.IMREAD_UNCHANGED)
#overlay = cv2.imread('wind_field_overlay.png', cv2.IMREAD_UNCHANGED)
#overlay = cv2.imread('wind_field_overlay.png')
#print(canvas.shape)
#print(overlay.shape)

canvas = overlay_transparent(canvas, overlay, 0, 0)

cv2.imshow("Game", canvas)
cv2.waitKey(8000)

'''
background = cv2.imread('field.jpg')
'''
