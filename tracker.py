from matplotlib import animation
import scipy.io as sio
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt

import solver

INPUT_PATH = "inputData/"
OUTPUT_PATH = "outputData/"
FILE = "tracking-cars.mat"


# ------ Preprocessor ------

# load video and set up coordinate system
mat_contents = sio.loadmat(INPUT_PATH + FILE)

# convert movie to double
movie = np.asarray(mat_contents['movie'], np.double)

# get first image (movie is a 3-d array, where the 3rd dimension is the frame number)
Is = movie[:, :, 0]

# generate coordinate system
[r, c] = Is.shape
x = list(range(0, c))
y = list(range(0, r))

# display image
plt.imshow(Is, cmap='gray')
plt.show()


# ------ Tracker ------

# coordinates of the middle point of the patch to track
# tr_x = [203 - 1]  # testing x coordinate for tracking-cars.mat
# tr_y = [110 - 1]  # testing y coordinate for tracking-cars.mat
tr_x = [int(input("Coordinate x: ")) - 1]
tr_y = [int(input("Coordinate y: ")) - 1]

# define 'radius' size of patch
t = 10

assert 0 < tr_x[0] < c and 0 < tr_y[0] < r

# set up patch coordinate system
[tx, ty] = np.meshgrid(range(-t, t + 1), range(-t, t + 1))
tx = tx.flatten()
ty = ty.flatten()

# loop over all images
lines = []
for i in range(1, movie.shape[2]):
    # get target image
    It = movie[:, :, i]

    # compute function and derivatives
    # compute gradient of source image, the result are two 'images'
    # storing d_xI_s and d_yI_s
    [dyIs, dxIs] = np.gradient(Is)

    # get all pixels of patch in d_xI_s
    # and store it in a variable called A1.
    # Note that we have to take the current endpoint
    # of the trajectory (tr_x[-1],tr_y[-1]).
    A1 = ndimage.map_coordinates(dxIs.T, (tr_x[-1] + tx, tr_y[-1] + ty), order=2)

    # get all pixels of patch in d_yI_s
    # and store it in a variable called A2.
    A2 = ndimage.map_coordinates(dyIs.T, (tr_x[-1] + tx, tr_y[-1] + ty), order=2)

    # get all pixels of patch in I_t - I_s
    # and store it in a variable called b.
    b = ndimage.map_coordinates(It.T, (tr_x[-1] + tx, tr_y[-1] + ty), order=2) - \
        ndimage.map_coordinates(Is.T, (tr_x[-1] + tx, tr_y[-1] + ty), order=2)

    # set up equation system
    # for that transform A1, A2 and b to column vectors
    A = np.stack((A1.T.flatten(), A2.T.flatten()), axis=-1)
    b = b.flatten()

    # solve equation system using LSP-solver
    [v, res] = solver.solve_lsp(A, b)

    # update trajectory
    # Attention! I_s(x+v) ~ I_t(x)
    # since we show always It, we have to subtract v !
    tr_x.append(tr_x[-1] - v[0])
    tr_y.append(tr_y[-1] - v[1])
    lines.append((np.asarray(tr_x), np.asarray(tr_y)))

    # update Is
    Is = It

# plot tracking output
fig = plt.figure()
ax = plt.axes()

line, = ax.plot([], [], lw=1)
img = ax.imshow(movie[:, :, 0], cmap='gray')


# Initialization function: plot the background of each frame
def init():
    line.set_data(lines[0][0], lines[0][1])
    img.set_data(movie[:, :, 0])
    return line,


# Animation function which updates figure data.  This is called sequentially
def animate(i):
    line.set_data(lines[i][0], lines[i][1])
    img.set_data(movie[:, :, i])
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(lines), interval=50)

# Call function to display the animation
anim.save(OUTPUT_PATH + "output.gif")
