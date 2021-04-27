import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import cv2


def region_seg(image, init_mask, max_iterations=350):
    alpha = 0.2
    display = True
    eps = np.finfo(float).eps
    # Convert image to gray uint8 type and apply 0.5 gamma correction
    gray_image = np.uint8((cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) * 255))

    fig, axes = plt.subplots(ncols=1)

    # Create a signed distance map (SDF) from initial mask
    phi = mask2phi(init_mask)

    # Main loop
    its = 0
    terminate_convergence = False
    prev_mask = init_mask
    c = 0
    thresh = 0;
    while its < max_iterations and not terminate_convergence:
        # Get the curve's narrow band
        idx = np.flatnonzero(np.logical_and(phi <= 1.2, phi >= -1.2))

        if len(idx) > 0:
            if display:
                if np.mod(its, 50) == 0:
                    print('iteration: {0}'.format(its))
                    animate_curve_and_phi(fig, image, phi)
            else:
                if np.mod(its, 10) == 0:
                    print('iteration: {0}'.format(its))

            # Find interior and exterior mean
            upts = np.flatnonzero(phi <= 0)  # interior points
            vpts = np.flatnonzero(phi > 0)  # exterior points
            u = np.sum(gray_image.flat[upts]) / (len(upts) + eps)  # interior mean
            v = np.sum(gray_image.flat[vpts]) / (len(vpts) + eps)  # exterior mean

            # Force from image information
            f = (gray_image.flat[idx] - u) ** 2 - (gray_image.flat[idx] - v) ** 2
            # Force from curvature penalty
            curvature = get_curvature(phi, idx)

            # SMOOTH BORDER - CHECK PAPER

            # Enforcing curve stability by maximum smoothing constraint on
            # Courant-Friedreichs-Lewy (CFL) function. Speed distance function
            # and maximum smoothing constraints improve CFL, evolving curves in a
            # smoother manner and reducing iterations required.
            dphidt = f / np.max(np.abs(f)) + alpha * curvature
            speed_distance = np.abs((255 - phi.flat[idx] + u) / v)
            derivation = speed_distance / (max(dphidt) + eps)

            # Evolve the curve
            phi.flat[idx] = phi.flat[idx] + derivation * dphidt

            # Keep SDF smooth
            phi = sussman(phi, 0.5)

            new_mask = phi <= 0
            c = convergence(prev_mask, new_mask, thresh, c)

            # check whether curve has evolved, if not, terminate
            if c <= 5:
                its = its + 1
                prev_mask = new_mask
            else:
                terminate_convergence = True

        else:
            break

    # Final output
    if display:
        animate_curve_and_phi(fig, image, phi)
        plt.savefig('segmented_lesion.png', bbox_inches='tight')

    # Make mask from SDF
    seg = phi <= 0  # Get mask from levelset

    return seg


# -----AUXILIARY FUNCTIONS-----#

# Convert mask to signed distance map (SDF)
def mask2phi(init_mask):
    phi = bwdist(init_mask) - bwdist(1 - init_mask) + im2double(init_mask) - 0.5
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


def im2double(image):
    image = image.astype(np.float32)
    image /= np.abs(image).max()
    return image


# Compute Euclidean distance transform of mask/binary image (bw)
def bwdist(a):
    return nd.distance_transform_edt(a == 0)


# Displays the image with curve superimposed
def animate_curve_and_phi(fig, image, phi):
    fig.axes[0].cla()
    fig.axes[0].imshow(image, 'gray')  # change to display coloured image
    fig.axes[0].contour(phi, 0, colors='r')
    fig.axes[0].set_axis_off()
    fig.axes[0].set_title("Convergence")
    plt.draw()

    plt.pause(0.001)


def sussman_sign(D):
    return D / np.sqrt(D ** 2 + 1)


# Check change in convergence
def convergence(previous_mask, current_mask, thresh, c):
    diff = np.subtract(previous_mask, current_mask, dtype=np.float32)
    n_diff = np.sum(np.abs(diff))
    if n_diff < thresh:
        c = c + 1
    else:
        c = 0
    return c
