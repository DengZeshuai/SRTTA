import numpy as np
from scipy import special
from scipy import ndimage
from skimage.filters import gaussian
import random
import cv2
import torch
import torch.nn.functional as F

import utils.utils_image as util
import utils.utils_blindsr as blindsr
import utils.basicsr_degradations as degradations


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter, ref: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter

    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
    

def add_sinc_blur(img):
    """
    Add ringing and overshoot artifacts to the image, using 2D sinc kernel
    params: 
        img: HxWxC, [0, 1]
    """
    kernel_range = [2 * v + 1 for v in range(3, 11)]
    kernel_size = random.choice(kernel_range)
    
    if kernel_size < 13:
        omega_c = np.random.uniform(np.pi / 3, np.pi)
    else:
        omega_c = np.random.uniform(np.pi / 5, np.pi)
    kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)

    # convert to tensor
    kernel = torch.FloatTensor(kernel)
    img = util.single2tensor4(img)

    img_out = filter2D(img, kernel)
    img_out = torch.clamp(img_out, 0, 1)

    # convert to numpy
    img_out = util.tensor2single3(img_out)

    return img_out


def add_resize_blur(img, scale_range=2, prob=0.5):
    """Randown Upsampling and Downsampling
    params: 
        img: HxWxC, [0, 1]
    """
    h, w = img.shape[:2]
    
    if np.random.random() > 0.5:  
        # random upsampling
        up_sf = random.uniform(1.1, scale_range)
        # upsampling downsampling with random interpolation
        img = cv2.resize(img, (int(up_sf*w), int(up_sf*h)), interpolation=random.choice([1, 2, 3]))
    
    # random downsampling
    down_sf = random.uniform(1/scale_range, 0.9)
    
    # downsampling with random interpolation
    img = cv2.resize(img, (int(down_sf*w), int(down_sf*h)), interpolation=random.choice([1, 2, 3]))

    # resize back to original size
    img = cv2.resize(img, (w, h), interpolation=random.choice([1, 2, 3]))

    img = np.clip(img, 0.0, 1.0)

    return img


def gaussian_blur(img, scale=2):
    wd = 2.0 + 0.2*scale
    # iso gaussian blur with random size and sigma
    k = blindsr.fspecial('gaussian', 2*random.randint(2,11)+3, wd*random.random())
    img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')
    return img


def add_glass_blur(img, scale=2):
    # wd = 2.0 + 0.2*scale
    # rand_sigma = random.random() * wd
    rand_sigma = random.uniform(1, 2)
    rand_range = random.randint(2, 6)
    rand_iter = random.randint(2, 3)
    # round and clip image for counting vals correctly
    # img = gaussian(img, sigma=rand_sigma, channel_axis=True)
    img = gaussian(img, sigma=rand_sigma)
    # img = np.clip((img * 255.0).round(), 0, 255) / 255.

    h, w = img.shape[:2]
    # locally shuffle pixels
    for _ in range(rand_iter):
        for h in range(h - rand_range, rand_range, -1):
            for w in range(w - rand_range, rand_range, -1):
                # random raletive position
                dx, dy = np.random.randint(-rand_range, rand_range, size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap pixel values
                img[h, w], img[h_prime, w_prime] = img[h_prime, w_prime], img[h, w]

    # round and clip image for counting vals correctly
    # img = gaussian_blur(img, scale)
    # img = gaussian(img, sigma=rand_sigma, channel_axis=True)
    img = gaussian(img, sigma=rand_sigma)
    img = np.clip(img, 0, 1)

    return img


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur) # TODO: test this operation do not need random


def add_defocus_blur(img):
    rand_radius = random.randint(3, 10)
    rand_alias_blur = np.random.uniform(0.1, 0.5)

    kernel = disk(radius=rand_radius, alias_blur=rand_alias_blur)
    img = cv2.filter2D(img, -1, kernel)
    img = np.clip(img, 0, 1)

    return img


# avoid the scale to be too small to gennerate test data
def add_scale_Poisson_noise(img, scale_range=(0.5, 3), gray_prob=0.5):
    """
     params: 
        img: HxWxC, [0, 1]
    """
    img = degradations.random_add_poisson_noise(
        img, scale_range, gray_prob, clip=True, rounds=False)

    return img


def _bernoulli(p, shape):
    """
    https://github.com/scikit-image/scikit-image/blob/main/skimage/util/noise.py#L191
    
    Bernoulli trials at a given probability of a given size.
    This function is meant as a lower-memory alternative to calls such as
    `np.random.choice([True, False], size=image.shape, p=[p, 1-p])`.
    While `np.random.choice` can handle many classes, for the 2-class case
    (Bernoulli trials), this function is much more efficient.
    Parameters
    ----------
    p : float
        The probability that any given trial returns `True`.
    shape : int or tuple of ints
        The shape of the ndarray to return.

    Returns
    -------
    out : ndarray[bool]
        The results of Bernoulli trials in the given `size` where success
        occurs with probability `p`.
    """
    if p == 0:
        return np.zeros(shape, dtype=bool)
    if p == 1:
        return np.ones(shape, dtype=bool)
    return np.random.random(shape) <= p


def add_impluse_noise(img, noise_prob=(0.03, 0.25), salt_vs_pepper_prob=(0, 1), gray_prob=0.5):
    """
     params: 
        img: HxWxC, [0, 1]
    """
    img_out = img.copy()
    
    rand_prob = np.random.uniform(noise_prob[0], noise_prob[1])
    salt_vs_pepper = np.random.uniform(salt_vs_pepper_prob[0], salt_vs_pepper_prob[1])
    
    if np.random.random() < gray_prob:
        flipped = _bernoulli(rand_prob, (*img.shape[:2], 1))
        salted = _bernoulli(salt_vs_pepper, (*img.shape[:2], 1))
        flipped = np.repeat(flipped, 3, axis=2)
        salted = np.repeat(salted, 3, axis=2)
        peppered = ~salted
        img_out[flipped & salted] = 1.
        img_out[flipped & peppered] = 0
    else:
        # flipped = np.random.binomial(n=1, p=rand_prob, size=img.shape)
        # salted = np.random.binomial(n=1, p=salt_vs_pepper, size=img.shape)
        flipped = _bernoulli(rand_prob, img.shape)
        salted = _bernoulli(salt_vs_pepper, img.shape)
        peppered = ~salted
        img_out[flipped & salted] = 1.
        img_out[flipped & peppered] = 0
    
    return img_out
