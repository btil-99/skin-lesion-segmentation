import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import cv2

"""
This code implements the paper: "Active Contours Without
Edges" By Chan Vese. 

Implementation was taken from Shawn Lankton [1] and translated into Python code


Reference:
[1] Shawn Lankton (2021). Active Contour Segmentation 
(https://www.mathworks.com/matlabcentral/fileexchange/19567-active-contour-segmentation), 
MATLAB Central File Exchange. Retrieved April 27, 2021.

"""


def region_seg(image, init_mask, max_iterations=250):
    alpha = 0.2
    eps = np.finfo(float).eps
    # Convert image to gray uint8 type
    gray_image = np.uint8((cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) * 255))

    fig, axes = plt.subplots(ncols=1)

    # Create a signed distance map (SDF) from initial mask
    phi = mask2phi(init_mask)

    # Main loop
    iterations = 0

    while iterations < max_iterations:
        # Get the curve's narrow band
        idx = np.flatnonzero(np.logical_and(phi <= 1.2, phi >= -1.2))

        # Display curve animation
        if len(idx) > 0:
            if np.mod(iterations, 10) == 0:
                print('iteration: {0}'.format(iterations))
                show_curve(fig, image, phi)

            # Find interior and exterior mean
            upts = np.flatnonzero(phi <= 0)  # interior points
            vpts = np.flatnonzero(phi > 0)  # exterior points
            u = np.sum(gray_image.flat[upts]) / (len(upts) + eps)  # interior mean
            v = np.sum(gray_image.flat[vpts]) / (len(vpts) + eps)  # exterior mean

            # Force from image information
            f = (gray_image.flat[idx] - u) ** 2 - (gray_image.flat[idx] - v) ** 2
            # Get forces from curvature penalty
            curvature = get_curvature(phi, idx)

            # Enforcing curve stability by maximum smoothing constraint on
            # Courant-Friedreichs-Lewy (CFL) function. Speed distance function
            # and maximum smoothing constraints improve CFL, evolving curves in a
            # smoother manner and reducing iterations required.
            dphidt = f / np.max(np.abs(f)) + alpha * curvature # gradient descent to minimise energy
            speed_distance = np.abs((255 - phi.flat[idx] + u) / v)
            derivation = speed_distance / (max(dphidt) + eps)

            # Evolve the curve
            phi.flat[idx] = phi.flat[idx] + derivation * dphidt

            # Keep SDF smooth
            phi = sussman(phi, 0.5)
            iterations = iterations + 1
        else:
            break

    # Final output
    show_curve(fig, image, phi)

    # Make mask from SDF
    seg = phi <= 0  # Get mask from levelset
    plt.close()
    return seg


# -----AUXILIARY FUNCTIONS-----#

# Displays the image with curve superimposed
def show_curve(fig, image, phi):
    fig.axes[0].cla()
    fig.axes[0].imshow(image)  # change to display coloured image
    fig.axes[0].contour(phi, 0, colors='r')
    fig.axes[0].set_axis_off()
    fig.axes[0].set_title("Convergence")
    plt.draw()

    plt.pause(0.0001)


# Convert mask to signed distance map (SDF)
# Calculated by subtracting euclidean distance transform (EDT) of the inverted initial mask from the
# EDT of original initial mask
def mask2phi(init_mask):
    phi = distance_transform(init_mask) - distance_transform(1 - init_mask)
    return phi


# Compute curvature along SDF
def get_curvature(phi, idx):
    dimy, dimx = phi.shape
    yx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # Get subscripts
    y = yx[:, 0]
    x = yx[:, 1]

    # Get subscripts of neighbors
    ym1 = y - 1
    xm1 = x - 1
    yp1 = y + 1
    xp1 = x + 1

    # Bounds checking
    ym1[ym1 < 0] = 0
    xm1[xm1 < 0] = 0
    yp1[yp1 >= dimy] = dimy - 1
    xp1[xp1 >= dimx] = dimx - 1

    # Get indexes for 8 neighbors
    idup = np.ravel_multi_index((yp1, x), phi.shape)
    iddn = np.ravel_multi_index((ym1, x), phi.shape)
    idlt = np.ravel_multi_index((y, xm1), phi.shape)
    idrt = np.ravel_multi_index((y, xp1), phi.shape)
    idul = np.ravel_multi_index((yp1, xm1), phi.shape)
    idur = np.ravel_multi_index((yp1, xp1), phi.shape)
    iddl = np.ravel_multi_index((ym1, xm1), phi.shape)
    iddr = np.ravel_multi_index((ym1, xp1), phi.shape)

    # Get central derivatives of SDF at x,y
    phi_x = -phi.flat[idlt] + phi.flat[idrt]
    phi_y = -phi.flat[iddn] + phi.flat[idup]
    phi_xx = phi.flat[idlt] - 2 * phi.flat[idx] + phi.flat[idrt]
    phi_yy = phi.flat[iddn] - 2 * phi.flat[idx] + phi.flat[idup]
    phi_xy = -0.25 * phi.flat[iddl] - 0.25 * phi.flat[idur] + 0.25 * phi.flat[iddr] + 0.25 * phi.flat[idul]
    phi_x2 = phi_x ** 2
    phi_y2 = phi_y ** 2

    # Compute curvature (Kappa)
    curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
                 (phi_x2 + phi_y2 + np.finfo(float).eps) ** (3 / 2)) * (phi_x2 + phi_y2) ** 0.5

    return curvature


def im2double(image):
    image = image.astype(np.float32)
    image /= np.abs(image).max()
    return image


# Level set re-initialisation by the sussman method
def sussman(D, dt):
    # forward/backward differences
    a = D - np.roll(D, 1, axis=1)  # backward
    b = np.roll(D, -1, axis=1) - D  # forward
    c = D - np.roll(D, -1, axis=0)  # backward
    d = np.roll(D, 1, axis=0) - D  # forward

    a_p = np.clip(a, 0, np.inf)
    a_n = np.clip(a, -np.inf, 0)
    b_p = np.clip(b, 0, np.inf)
    b_n = np.clip(b, -np.inf, 0)
    c_p = np.clip(c, 0, np.inf)
    c_n = np.clip(c, -np.inf, 0)
    d_p = np.clip(d, 0, np.inf)
    d_n = np.clip(d, -np.inf, 0)

    a_p[a < 0] = 0
    a_n[a > 0] = 0
    b_p[b < 0] = 0
    b_n[b > 0] = 0
    c_p[c < 0] = 0
    c_n[c > 0] = 0
    d_p[d < 0] = 0
    d_n[d > 0] = 0

    dD = np.zeros(D.shape)
    D_neg_ind = np.flatnonzero(D < 0)
    D_pos_ind = np.flatnonzero(D > 0)

    dD.flat[D_pos_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_p.flat[D_pos_ind] ** 2], [b_n.flat[D_pos_ind] ** 2])), axis=0) +
        np.max(np.concatenate(
            ([c_p.flat[D_pos_ind] ** 2], [d_n.flat[D_pos_ind] ** 2])), axis=0)) - 1

    dD.flat[D_neg_ind] = np.sqrt(
        np.max(np.concatenate(
            ([a_n.flat[D_neg_ind] ** 2], [b_p.flat[D_neg_ind] ** 2])), axis=0) +
        np.max(np.concatenate(
            ([c_n.flat[D_neg_ind] ** 2], [d_p.flat[D_neg_ind] ** 2])), axis=0)) - 1

    D = D - dt * sussman_sign(D) * dD

    return D


def sussman_sign(D):
    return D / np.sqrt(D ** 2 + 1)


# Compute Euclidean distance transform of mask/binary image (bw)
def distance_transform(mask):
    return nd.distance_transform_edt(mask == 0)
