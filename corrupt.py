import sys
sys.path.append("../")
from utils import transform_image_to_kspace, transform_kspace_to_image
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from gstools import SRF, Gaussian
from scipy.fftpack import dct, idct

# functions for downsample

def cartesian_mask(factor, PE_direction, distribution):
    acceleration, center_fraction = factor
    size = 320
    mask = np.zeros((size, size), dtype=bool)
    num_cols = size
    num_low_frequencies = round(num_cols * center_fraction)
       
    if (distribution == "uniform"):
        adjusted_accel = round((acceleration * (num_low_frequencies - num_cols)) / (num_low_frequencies * acceleration - num_cols))
        offset = np.random.randint(0, round(adjusted_accel))
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
    elif (distribution == "random"):
        prob = (num_cols / acceleration - num_low_frequencies) / (num_cols - num_low_frequencies)
        accel_samples = np.random.uniform(size=num_cols) < prob
    
    if (PE_direction == "LR"):
        mask[:, round((num_cols - num_low_frequencies - 2) / 2):round((num_cols + num_low_frequencies - 2) / 2)] = True
        mask[:, accel_samples] = True
    elif (PE_direction == "AP"):
        mask[round((num_cols - num_low_frequencies - 2) / 2):round((num_cols + num_low_frequencies - 2) / 2), :] = True
        mask[accel_samples, :] = True
 
    return mask

# functions for motion

def c2h(X):
    # convert cartesian to homogeneous coordinates
    
    n = np.ones([1,X.shape[1]])
    Xh = np.concatenate((X,n))

    return Xh

def t2h(T, t):
    # convert a 2D transformation matrix and 2D translation vector to homogeneous transformation matrix

    T1 = np.concatenate((T, t[:,None]), axis=1)
    n = np.zeros([1, T1.shape[1]])
    n[-1,-1] = 1
    Th = np.concatenate((T1, n), axis=0)
    
    return Th

def rotate(phi):
    # create a 2D rotation matrix
    
    T = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    
    return T

def image_transform(image, Th):
    # image transformation by inverse mapping

    # spatial coordinates of the transformed image
    x = np.arange(0, image.shape[1])
    y = np.arange(0, image.shape[0])
    xx, yy = np.meshgrid(x, y)

    # convert to a 2-by-p matrix (p is the number of pixels)
    X = np.concatenate((xx.reshape((1, xx.size)), yy.reshape((1, yy.size))))
    # convert to homogeneous coordinates
    Xh = c2h(X)

    # perform inverse coordinates mapping
    T_inv = np.linalg.pinv(Th)
    Xt = T_inv.dot(Xh) 
    image_t = ndimage.map_coordinates(image, [Xt[1,:], Xt[0,:]], order=3, mode='constant', cval=0.0).reshape(image.shape)

    return image_t

def combining_transforms(image, ang_deg, trans_x, trans_y):

    # rotation around the image center
    ang_rad = np.deg2rad(ang_deg)
    T_1 = t2h(np.eye(2), np.array([int(image.shape[1]/2), int(image.shape[0]/2)]))
    T_2 = t2h(rotate(ang_rad), np.zeros(2))
    T_3 = t2h(np.eye(2), np.array([-int(image.shape[1]/2), -int(image.shape[0]/2)]))
    T_rot = T_1.dot(T_2).dot(T_3)

    # translation
    T_trans = t2h(np.eye(2), np.array([trans_x, trans_y]))
    
    # combine transforms
    T_tot = T_trans.dot(T_rot)
    
    image_t = image_transform(image, T_tot)
    
    return image_t

def rigid_motion(image, pe_direction, n_movements, ang_std, trans_std):
    # normalize the image from 0 to 1
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # locations of the movements
    if (pe_direction == "LR"):
        grid_size = image.shape[1]
    elif (pe_direction == "AP"):
        grid_size = image.shape[0]
    locs = np.sort(np.append(np.random.permutation(grid_size)[:(2*n_movements)], (0, grid_size)))

    # rotation and translation parameters
    ang = np.random.normal(0, ang_std, n_movements)
    trans_x = np.random.normal(0, trans_std, n_movements)
    trans_y = np.random.normal(0, trans_std, n_movements)

    # combine kspaces
    kspace = transform_image_to_kspace(image)
    for i in range(n_movements):
        img_i = combining_transforms(image, sum(ang[:(i+1)]), sum(trans_x[:(i+1)]), sum(trans_y[:(i+1)]))
        kspace_i = transform_image_to_kspace(img_i)
        if (pe_direction == "LR"):
            kspace[:, locs[2*i+1]:locs[2*i+2]] = kspace_i[:, locs[2*i+1]:locs[2*i+2]]
        elif (pe_direction == "AP"):
            kspace[locs[2*i+1]:locs[2*i+2], :] = kspace_i[locs[2*i+1]:locs[2*i+2], :]
    
    return kspace

def periodic_motion(kspace):
    x = np.linspace(-np.pi, np.pi, kspace.shape[1])
    y = np.linspace(-np.pi, np.pi, kspace.shape[0])
    kx, ky = np.meshgrid(x, y)

    # parameters
    alpha = np.random.uniform(0.1, 5) # respiratory frequency
    delta = np.random.uniform(0, 20) # shift along PE direction
    beta = np.random.uniform(0, np.pi/4) # phase
    ky0 = np.random.uniform(np.pi/10, np.pi/2) # center K-space lines without phase shift errors
    
    # phase error outside of center
    phase_error = ky*delta*np.sin(alpha*ky + beta)
    
    # no motion in the center of k-space
    phase_error[abs(ky) < ky0] = 0

    # add phase error to k-space
    kspace = kspace*np.exp(-1j*phase_error)

    return kspace


# 2D discrete cosine transform
def dct2(grid):   
    # discrete cosine transform
    M = grid.shape[0]
    N = grid.shape[1]
    a = np.empty([M,M],float)
    b = np.empty([M,M],float)
    for i in range(M):
        a[i,:] = dct(grid[i,:], norm='ortho')
    for j in range(N):
        b[:,j] = dct(a[:,j], norm='ortho')
    
    # keep essential DCT coefficients
    b = b[:M, :N]
    
    # inverse discrete cosine transform
    m = b.shape[0]
    n = b.shape[1]
    M = 320
    N = 320
    a = np.empty([m,N],float)
    grid = np.empty([M,N],float)
    for i in range(m):
        a[i,:] = idct(b[i,:], n=M, norm='ortho')
    for j in range(N):
        grid[:,j] = idct(a[:,j], n=N, norm='ortho')
        
    return grid

def GetFields(image):    
    down_size = np.int32(np.divide(image.shape, 10))
    x = np.arange(down_size[0])
    y = np.arange(down_size[1])
    
    len_scale = np.random.uniform(10, 50)
    model = Gaussian(dim=2, var=50, len_scale=len_scale)

    srf = SRF(model)

    grid_z = srf((x, y), mesh_type='structured')
    grid_z = dct2(grid_z)
    
    bias_rng = np.random.uniform(0.20, 1.00)
    grid_z = np.interp(grid_z, (grid_z.min(), grid_z.max()),
                       (1 - bias_rng / 2, 1 + bias_rng / 2))
    
    return grid_z

def corrupt_image(image, case):
    import random
    kspace = transform_image_to_kspace(image)
    
    if case == "noise":
        signal_to_noise = np.random.uniform(0, 10)
        mean_signal = np.mean(np.abs(kspace))
        std_noise = mean_signal / 10**(signal_to_noise / 20)
        noise = np.random.normal(0, std_noise, size=np.shape(kspace)) + 1j*np.random.normal(0, std_noise, size=np.shape(kspace))
        kspace += noise
    
    elif case == "downsample":
        mask = cartesian_mask(random.choice([(2, 0.16), (3, 0.12), (4, 0.08)]), random.choice(["LR", "AP"]), random.choice(["uniform", "random"]))  
        kspace = np.where(mask, kspace, (0 + 0j))
        
    elif case == "motion_rigid":
        motion_type = "rigid", "periodic"
        if (motion_type == "rigid"):
            n_movements = np.random.randint(1, 6)
            ang_std = 0.6
            trans_std = 1.1
            pe_direction = random.choice(["LR", "AP"])
            kspace = rigid_motion(image, pe_direction, n_movements, ang_std, trans_std)
        elif (motion_type == "periodic"):
            kspace = periodic_motion(kspace)
            
    elif case == "motion":
        motion_type = random.choice(["rigid", "periodic"])
        if (motion_type == "rigid"):
            n_movements = np.random.randint(1, 6)
            ang_std = 0.6
            trans_std = 1.1
            pe_direction = random.choice(["LR", "AP"])
            kspace = rigid_motion(image, pe_direction, n_movements, ang_std, trans_std)
        elif (motion_type == "periodic"):
            kspace = periodic_motion(kspace)
            
    elif case == "bias":
        bias = GetFields(image)
        img = np.multiply(image, bias)
        kspace = transform_image_to_kspace(img)
      
    img = transform_kspace_to_image(kspace)
    img = (img - np.mean(img)) / np.std(img)
    return img