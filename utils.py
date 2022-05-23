# -*- coding: utf-8 -*-
"""
Contains helper functions and utility functions for use in the library.

Created on Mon Oct  9 14:05:26 2017

Copyright (c) 2017, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
__all__ = ["running_mean", "mse", "save_state", "Timer", "ReflectPadding2D", "MyCustomWeightShifter"]
# from MLTK.accelerated_mri.utils import radial_mask, frequency_mask, transform_image_to_kspace, transform_kspace_to_image, compare_k, corrupt
import tensorflow
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import time
import tensorflow
from tensorflow.keras import backend as K
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.signal
import gc
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.constraints import Constraint
import logging
import tensorflow as tf
# from typeguard import typechecked
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
# import cv2
from matplotlib import cm
from tensorflow.keras.models import Model
import matplotlib
# matplotlib.use('Agg')
from tensorflow.keras.layers import Lambda, Activation, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
from tensorflow.keras.models import Model
from scipy import ndimage
from sewar.full_ref import uqi, vifp

def radial_mask(thr):
    from skimage.draw import line
    size = 320
    mask = np.zeros((size, size), dtype=np.bool)
    for i in range(0, size + 1, thr):
        img = np.zeros((size, size), dtype=np.float32)
        nmb = i
        for j in range(2):
            if (j == 1):
                rr, cc = line(0, nmb, size, size - nmb)
            else:
                rr, cc = line(nmb, 0, size - nmb, size)

            cc = cc[rr < size]
            rr = rr[rr < size]

            rr = rr[cc < size]
            cc = cc[cc < size]
            img[rr, cc] = 1
            mask = mask | (img > 0)
    return mask


def frequency_mask(thr, horizontal, extra):
    from skimage.draw import line
    size = 320
    mask = np.zeros((size, size), dtype=np.bool)
    band = int(320 / thr)

    if (horizontal):
        mask[:, int((size - band - 2) / 2):int((size + band - 2) / 2)] = True
    else:
        mask[int((size - band - 2) / 2):int((size + band - 2) / 2), :] = True

    if (extra):
        for i in range(int(size / thr)):
            if (horizontal):
                mask[:, np.random.randint(0, 320)] = True
            else:
                mask[np.random.randint(0, 320), :] = True

            

    return mask

def normalize_z(tensor):
    import tensorflow.keras.backend as K
    t_mean = K.mean(tensor, axis=(1, 2))
    t_std = K.std(tensor, axis=(1, 2))
    return tf.math.divide_no_nan(tensor - t_mean[:, None, None, :], t_std[:, None, None, :])

def normalize_k(tensor):
    import tensorflow.keras.backend as K
    t_mean = K.max(K.abs(tensor), axis=(1, 2, 3))
    return tf.math.divide_no_nan(tensor, t_mean[:, None, None, None])


def img_to_kspace(tensor):
    from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
    dim = (1, 2)

    inp_c = tf.dtypes.complex(tensor[:, :, :, 0], tensor[:, :, :, 1])

    kspace = ifftshift(fft2d(fftshift(inp_c, axes=dim)), axes=dim)
    # kspace = tf.expand_dims(ifft2d(inp_c), 3)
    return tf.stack((tf.math.real(kspace), tf.math.imag(kspace)), 3) 

def kspace_to_img(tensor):
    from tensorflow.signal import fft2d, ifft2d, fftshift, ifftshift
    from tensorflow.keras.layers import Activation
    dim = (1, 2)

    inp_c = tf.dtypes.complex(tensor[:, :, :, 0], tensor[:, :, :, 1])
    img = fftshift(ifft2d(ifftshift(inp_c, axes=dim)), axes=dim)
    #img = tf.expand_dims(fft2d(inp_c), 3)
    return tf.stack((tf.math.real(img), tf.math.imag(img)), 3) 


def IQM(model, gen, in_size, comet):
    e_loss = []
    k_loss = []
    k_l_loss = []
    k_h_loss = []
    ssim_loss = []
    uqi_loss = []
    vif_loss = []
        
    for i in range(len(gen)):
        one, two = gen[i]

        pred = model.predict_on_batch(two[0])
        pred_unit = model.predict_on_batch(one[0])
        # pred = two[0] + diff

        # k_loss.append(compare_k(one[0], pred, two[0]))
        e_loss.append(compare_mse(one[0], pred))
        ssim_loss.append(compare_ssim(one[0], pred))

        one = one[0][0, :, :, 0]
        pred = pred[0, :, :, 0]
        pred_unit = pred_unit[0, :, :, 0]
        # one = np.interp(one, (np.min(one), np.mean(one)), (0, 1))
        # pred = np.interp(pred, (np.min(pred), np.mean(pred)), (0, 1))
        # pred_unit = np.interp(pred_unit, (np.min(pred_unit), np.mean(pred_unit)), (0, 1))
        uqi_loss.append(uqi(one, pred))
        vif_loss.append(vifp(one, pred))
        k_loss.append(vifp(one, pred_unit))
        
    comet.log_metrics({"MSE_test":round(np.mean(e_loss), 10),
                       "SSIM_test":round(np.mean(ssim_loss), 10),
                       "UQI_test":round(np.mean(uqi_loss), 10),
                       "VIF_test":round(np.mean(vif_loss), 10),
                       "VIFk_test":round(np.mean(k_loss), 10)})

    print("MSE: " + str(np.mean(e_loss)) + " +- " + str(np.std(e_loss)))
    print("SSIM: " + str(np.mean(ssim_loss)) + " +- " + str(np.std(ssim_loss)))
    print("UQI: " + str(np.mean(uqi_loss)) + " +- " + str(np.std(uqi_loss)))
    print("VIF: " + str(np.mean(vif_loss)) + " +- " + str(np.std(vif_loss)))
    print("VIF k: " + str(np.mean(k_loss)) + " +- " + str(np.std(k_loss)))


def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    # img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    # img = np.stack([np.real(img), np.imag(img)], 2)
    # img = (img - np.mean(img)) / np.std(img)
    return np.real(img)

def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    # img = np.interp(img, (np.min(img), np.max(img)), (0, 1))
    k = ifftshift(fftn(fftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

def sobel(img):
    sobel_img = []
    kernel_top = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_left = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    for i in range(np.shape(img)[0]):
        sobel_img.extend(np.expand_dims(np.sqrt(np.square(convolve2D(img[i, :, :, 0], kernel_top)) + np.square(convolve2D(img[i, :, :, 0], kernel_left))), 0))
    sobel_img = np.array(sobel_img)
    return sobel_img

def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def compare_kernel(img1, img2):
    pred1 = sobel(img1)
    pred2 = sobel(img2)

    return np.mean(np.square(pred1 - pred2), axis=(1, 2))

def compare_k(img1_full, img2_full, img_in):
    from sewar.full_ref import vifp


    loss = []
    for i in range(np.shape(img1_full)[0]):
        img1 = img1_full[i, :, :, 0]
        img2 = img2_full[i, :, :, 0]
        img_lr = transform_kspace_to_image(transform_image_to_kspace(img_in[i, :, :, 0]))
        mask = np.abs(transform_image_to_kspace(img_lr)) > 1e-5
        # mask = np.zeros((256, 256), dtype=bool)
        # mask[int((256 / 2) - (size / 2)) : int((256 / 2) + (size / 2)), int((256 / 2) - (size / 2)) : int((256 / 2) + (size / 2))] = True

        kspace_hr = np.where(mask, transform_image_to_kspace(img1), (0 + 0j))
        kspace_lr = np.where(mask, transform_image_to_kspace(img2), (0 + 0j))
        img1 = transform_kspace_to_image(kspace_hr)
        img2 = transform_kspace_to_image(kspace_lr)

        # img1 = np.interp(img1, (np.mean(img1), np.std(img1)), (0, 1))
        # img2 = np.interp(img2, (np.mean(img2), np.std(img2)), (0, 1))
        loss.append(np.mean(np.square(img_lr - img2)))
    

    return loss

def corrupt(image, case):
    import random
    kspace = transform_image_to_kspace(image)

    if case == "downsample":
        approach = random.choice(["cartesian", "radial"])
        if (approach == "cartesian"):
            mask = frequency_mask(random.choice([2, 4, 8]), random.choice([True, False]), random.choice([False]))
        elif (approach == "radial"):
            mask = radial_mask(random.choice([2, 4, 8]))

        kspace = np.where(mask, kspace, (0 + 0j))
    elif case == "noise":
        mag = np.random.uniform(0.1, 0.2)
        img = transform_kspace_to_image(kspace)
        img = (img - np.mean(img)) / np.std(img)
        kspace = transform_image_to_kspace(img + np.random.normal(0, mag, size=np.shape(kspace)))
    elif case == "motion":
        mag = random.choice([5])
        k = np.zeros_like(transform_image_to_kspace(kspace))
        for i in range(np.shape(k)[0]):
            if ((i > 180) & (random.random() < (1 / mag))):
                kspace = np.roll(kspace, random.randint(-8, 8), 0)
            k[i, :] = kspace[i, :]
        kspace = k
    
    img = transform_kspace_to_image(kspace)
    img = (img - np.mean(img)) / np.std(img)

    return img#np.stack([np.real(kspace), np.imag(kspace)], 2)




def downsample(volume):
    radial_mask_2 = frequency_mask(2)
    radial_mask_4 = frequency_mask(4)
    radial_mask_8 = frequency_mask(8)
    kspace = transform_image_to_kspace(volume)

    one_k = np.zeros_like(kspace)
    two_k = np.zeros_like(kspace)
    four_k = np.zeros_like(kspace)
    eight_k = np.zeros_like(kspace)

    one = transform_kspace_to_image(kspace)
    two = transform_kspace_to_image(np.where(radial_mask_2, kspace, (0 + 0j)))
    four = transform_kspace_to_image(np.where(radial_mask_4, kspace, (0 + 0j)))
    eight = transform_kspace_to_image(np.where(radial_mask_8, kspace, (0 + 0j)))

    one = (one - np.mean(one))  / np.std(one)
    two = (two - np.mean(two))  / np.std(two)
    four = (four - np.mean(four))  / np.std(four)
    eight = (eight - np.mean(eight))  / np.std(eight)

    return [one, two, four, eight]

def znorm(img):
    return (img - np.mean(img)) / np.std(img)

def compare_mse(img1, img2):
    loss = []
    for i in range(np.shape(img1)[0]):
        loss.append(np.mean(np.square(img1[i, :, :, :] - img2[i, :, :, :])))
    return loss

def compare_ssim(img1, img2):
    img2 = img2
    loss = []
    for i in range(np.shape(img1)[0]):
        loss.append(ssim(img1[i, :, :, 0], img2[i, :, :, 0], data_range=np.max(img1[i, :, :, 0]) - np.min(img1[i, :, :, 0])))
    return loss

def compare_vif(img1, img2):
    loss = []
    for i in range(np.shape(img1)[0]):
        loss.append(vifp(img1[i, :, :, 0], img2[i, :, :, 0]))
    return loss

def relu_range(x):
    x = tensorflow.where(K.greater(x, 0), x, K.zeros_like(x))
    x = tensorflow.where(K.less(x, 1), x, K.ones_like(x))
    # x = tensorflow.where(K.greater(x, 0), x, K.zeros_like(x))
    # x = tensorflow.where(K.less(x, 1), x, K.ones_like(x))
    
    # mean = tensorflow.reduce_mean(x, [1, 2, 3])
    # stdev = tensorflow.math.reduce_std(x, [1, 2, 3])

    # x = x - mean[:, None, None, None]
    # x = x / stdev[:, None, None, None]
    return x

def save_progress(lr, hr, kiki, save_path, epoch, comet, idx):
    save_all = False
    kc = 4
    limit = 2e-3
    rel_limit = 0.5

    k1 = kiki.K1.predict_on_batch(lr)
    i1 = kiki.I1.predict_on_batch(k1)
    k2 = kiki.K2.predict_on_batch(i1)
    i2 = kiki.I2.predict_on_batch(k2)

    k1 = k1[0, :, :, 0]
    i1 = i1[0, :, :, 0]
    k2 = k2[0, :, :, 0]
    i2 = i2[0, :, :, 0]
    lr = lr[0][0, :, :, 0]
    hr = hr[0][0, :, :, 0]
    k_lr = np.abs(transform_image_to_kspace(lr))
    k_k1 = np.abs(transform_image_to_kspace(k1))
    k_i1 = np.abs(transform_image_to_kspace(i1))
    k_k2 = np.abs(transform_image_to_kspace(k2))
    k_i2 = np.abs(transform_image_to_kspace(i2))
    k_hr = np.abs(transform_image_to_kspace(hr))

    k_lr = k_lr / np.max(np.abs(k_lr))
    k_k1 = k_k1 / np.max(np.abs(k_k1))
    k_i1 = k_i1 / np.max(np.abs(k_i1))
    k_k2 = k_k2 / np.max(np.abs(k_k2))
    k_i2 = k_i2 / np.max(np.abs(k_i2))
    k_hr = k_hr / np.max(np.abs(k_hr))

    lr = (lr - np.mean(lr)) / np.std(lr)
    k1 = (k1 - np.mean(k1)) / np.std(k1)
    i1 = (i1 - np.mean(i1)) / np.std(i1)
    k2 = (k2 - np.mean(k2)) / np.std(k2)
    i2 = (i2 - np.mean(i2)) / np.std(i2)
    hr = (hr - np.mean(hr)) / np.std(hr)

    vmx = np.max(hr)
    vmn = np.min(hr)
    dmx = 0.5
    kmx = 0.01

    plt.figure(figsize=(40, 20))
    plt.subplot(3, 6, 1)
    plt.imshow(lr, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 2)
    plt.imshow(k1, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 3)
    plt.imshow(i1, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 4)
    plt.imshow(k2, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 5)
    plt.imshow(i2, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 6)
    plt.imshow(hr, interpolation='none', cmap='gist_gray', vmin=vmn, vmax=vmx)
    plt.colorbar()
    plt.axis('off')

    plt.subplot(3, 6, 7)
    plt.imshow(np.abs(lr - hr), interpolation='none', cmap='gist_gray', vmin=0, vmax=dmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 8)
    plt.imshow(np.abs(k1 - hr), interpolation='none', cmap='gist_gray', vmin=0, vmax=dmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 9)
    plt.imshow(np.abs(i1 - hr), interpolation='none', cmap='gist_gray', vmin=0, vmax=dmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 10)
    plt.imshow(np.abs(k2 - hr), interpolation='none', cmap='gist_gray', vmin=0, vmax=dmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 11)
    plt.imshow(np.abs(i2 - hr), interpolation='none', cmap='gist_gray', vmin=0, vmax=dmx)
    plt.colorbar()
    plt.axis('off')

    plt.subplot(3, 6, 13)
    plt.imshow(k_lr, interpolation='none', cmap='gist_gray', vmin=0, vmax=kmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 14)
    plt.imshow(k_k1, interpolation='none', cmap='gist_gray', vmin=0, vmax=kmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 15)
    plt.imshow(k_i1, interpolation='none', cmap='gist_gray', vmin=0, vmax=kmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 16)
    plt.imshow(k_k2, interpolation='none', cmap='gist_gray', vmin=0, vmax=kmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 17)
    plt.imshow(k_i2, interpolation='none', cmap='gist_gray', vmin=0, vmax=kmx)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3, 6, 18)
    plt.imshow(k_hr, interpolation='none', cmap='gist_gray', vmin=0, vmax=kmx)
    plt.colorbar()
    plt.axis('off')

    plt.savefig(save_path + "pics/" + str(idx) + ".svg")
    
    if (comet != None):
        if (save_all):
            comet.log_figure(figure=plt, figure_name=str(epoch) + "_" + str(idx), overwrite=True, step=epoch)
        else:
            comet.log_figure(figure=plt, figure_name=str(idx), overwrite=True, step=epoch)
    plt.close('all')

def rel_diff(img, gt):
    min_value = np.min(np.abs(gt[gt != 0]))
    return np.where(gt == 0, (img - gt) / min_value, (img - gt) / gt)

def get_slices(gen, idx):
    _, x = gen[idx]
    slice1 = x[0]
    slice2 = x[1]
    slice3 = x[2]
    # mean1 = np.mean(slice1, axis=(1,2,3))
    # mean2 = np.mean(slice2, axis=(1,2,3))
    # mean3 = np.mean(slice3, axis=(1,2,3))
    # std1 = np.std(slice1, axis=(1,2,3))
    # std2 = np.std(slice2, axis=(1,2,3))
    # std3 = np.std(slice3, axis=(1,2,3))
    # slice1 = (slice1 - mean1[:, None, None, None]) / std1[:, None, None, None]
    # slice2 = (slice2 - mean2[:, None, None, None]) / std2[:, None, None, None]
    # slice3 = (slice3 - mean3[:, None, None, None]) / std3[:, None, None, None]
    
    return slice1, slice2, slice3

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred))

def mse_neg(orig, pred):
    orig = tf.where(K.greater(orig, -10), orig, -10*K.ones_like(orig))
    orig = tf.where(K.less(orig, 10), orig, 10*K.ones_like(orig))
    pred = tf.where(K.greater(pred, -10), pred, -10*K.ones_like(pred))
    pred = tf.where(K.less(pred, 10), pred, 10*K.ones_like(pred))
    return -tf.reduce_mean(tf.keras.metrics.mean_squared_error(orig, pred))

def relu_range(x):

    # mean = tf.reduce_mean(x, [1, 2, 3])
    # stdev = tf.math.reduce_std(x, [1, 2, 3])

    # x = x - mean[:, None, None, None]
    # x = x / stdev[:, None, None, None]


    x = tf.where(K.greater(x, -10), x, -10*K.ones_like(x))
    # x = tf.where(K.greater(x, 0), x, K.zeros_like(x))
    x = tf.where(K.less(x, 10), x, 10*K.ones_like(x))
    return x

def build_compare(m_kernel):
    im1 = Input(shape=(256, 256, 1))
    im2 = Input(shape=(256, 256, 1))
    im3 = Input(shape=(256, 256, 1))

    # diff_im = tf.subtract(im1, im2)
    # diff_im = tf.square(diff_im)

    # diff = m_kernel(diff_im)[0]

    pred1 = m_kernel(im1)
    pred1 = tf.square(pred1)
    pred1 = tf.abs(tf.reduce_sum(pred1, axis=0))
    pred1 = tf.math.sqrt(tf.maximum(pred1, 1e-9))
    # for i in range(len(pred1)):
    #     pred1[i] = pred1[i] - tf.reduce_mean(pred1[i], [1, 2, 3])[:, None, None, None]
    #     pred1[i] = pred1[i] / tf.math.reduce_std(pred1[i], [1, 2, 3])[:, None, None, None]
    # pred1 = tf.square(pred1)
    # pred1 = tf.reduce_sum(pred1, axis=0)
    # pred1 = tf.sqrt(pred1)
    # pred1 = tf.divide(pred1, tf.reduce_max(pred1, [1, 2, 3])[:, None, None, None])
    # pred1 = Activation('relu')(pred1)
    # pred1 = Lambda(relu_range)(pred1)
    pred2 = m_kernel(im2)
    pred2 = tf.square(pred2)
    pred2 = tf.abs(tf.reduce_sum(pred2, axis=0))
    pred2 = tf.math.sqrt(tf.maximum(pred2, 1e-9))
    # for i in range(len(pred2)):
    #     pred2[i] = pred2[i] - tf.reduce_mean(pred2[i], [1, 2, 3])[:, None, None, None]
    #     pred2[i] = pred2[i] / tf.math.reduce_std(pred2[i], [1, 2, 3])[:, None, None, None]
    # pred2 = tf.square(pred2)
    # pred2 = tf.reduce_sum(pred2, axis=0)
    # pred2 = tf.sqrt(pred2)
    # pred2 = tf.divide(pred2, tf.maximum(pred2, [1, 2, 3])[:, None, None, None])
    # pred2 = Activation('relu')(pred2)
    # pred2 = Lambda(relu_range)(pred2)
    diff = tf.subtract(pred1, pred2)
    diff = Lambda(relu_range)(diff)
    # diff = tf.square(diff)
    # diff = tf.reduce_sum(diff, axis=0)
    # diff = tf.math.sqrt(diff)

    intersect = tf.reduce_max(tf.math.maximum(tf.abs(pred1), tf.abs(pred2))) + tf.reduce_mean(tf.math.maximum(tf.abs(pred1), tf.abs(pred2))) # tf.add(tf.abs(pred1), tf.abs(pred2))

    # diff = diff / tf.math.reduce_max(diff, [1, 2, 3])[:, None, None, None]
    # diff = diff - tf.reduce_mean(diff, [1, 2, 3])[:, None, None, None]

    # diff_min = tf.reduce_min(diff, axis=0)
    # diff_max = tf.reduce_max(diff, axis=0)
    # diff = tf.where(diff_max > tf.abs(diff_min), diff_max, diff_min)

    return Model(inputs=[im1, im2], outputs=[diff, intersect])

def build_full(m_base, m_kernel, size):
    from tensorflow.keras.layers import Lambda, Input, MaxPooling2D, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, SpatialDropout2D
    from tensorflow.keras.models import Model
    im1 = Input(shape=(512, 512, 1))
    dim = (1, 2)

    def sub_k(img):
        img = tf.math.divide(tf.subtract(img, tf.reduce_min(img, [1, 2])[:, None, None]), 
                             tf.subtract(tf.reduce_max(img, [1, 2])[:, None, None], tf.reduce_min(img, [1, 2])[:, None, None]))
    
        k = tensorflow.dtypes.complex(img, tensorflow.zeros_like(img))[:, :, :, 0]
        k = tensorflow.signal.ifftshift(k, axes=dim)
        k = tensorflow.signal.fft2d(k)
        k = tensorflow.signal.fftshift(k, axes=dim)
        k = k[:, int((512 / 2) - (size / 2)) : int((512 / 2) + (size / 2)), int((512 / 2) - (size / 2)) : int((512 / 2) + (size / 2))]
        k = tensorflow.pad(k, tf.constant([[0, 0], [int((512 - size) / 2), int((512 - size) / 2)], [int((512 - size) / 2), int((512 - size) / 2)]]), 'CONSTANT')
        # k = k * tf.complex64(tf.convert_to_tensor(mask[None, :, :, :]), tf.convert_to_tensor(mask[None, :, :, :]))

        k = tensorflow.signal.fftshift(k, axes=dim)
        k = tensorflow.signal.ifft2d(k)
        k = tensorflow.signal.ifftshift(k, axes=dim)
        img = tensorflow.abs(k)
        img = img - tensorflow.math.reduce_mean(img, [1, 2])[:, None, None]
        img = img / tensorflow.math.reduce_std(img, [1, 2])[:, None, None]
        return img

    m_kernel.trainable = False
    for layer in m_kernel.layers:
        layer.trainable = False

    pred = m_base([im1])

    pred_base = Lambda(sub_k)(pred)
    # pred_in = Lambda(sub_k)(im1)
    # reg = tf.keras.regularizers.L2(1.)
    # diff = tf.math.subtract(pred_base, pred_in)
    # reg_loss = reg(diff)
    

    pred_kernel = m_kernel(pred)
    model =  Model(inputs=[im1], outputs=[pred_base, pred_kernel])
    
    
    # m_kernel.trainable = True
    # for layer in m_kernel.layers:
    #     layer.trainable = True

    # im2_kernel = m_kernel(im2)
    # loss = tf.subtract(pred_kernel, im2_kernel)
    return model

def build_full_sr(m_base, m_kernel):
    im1 = Input(shape=(512, 512, 1))
    # im2 = Input(shape=(512, 512, 1))
    im3 = Input(shape=(512, 512, 1))

    pred = m_base(im1)

    # diff = tf.subtract(pred, im2)
    # loss = m_kernel(diff)

    pred_kernel = m_kernel(pred)
    # im2_kernel = m_kernel(im2)
    # loss = tf.subtract(pred_kernel, im2_kernel)
    return Model(inputs=[im1], outputs=[pred, pred_kernel])

if __name__ == "__main__":
    import doctest
    doctest.testmod()
