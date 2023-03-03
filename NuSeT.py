from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfile
from tkinter.ttk import Progressbar
import PIL.Image, PIL.ImageTk
from skimage.transform import rescale
import numpy as np
import os
import cv2
import tensorflow as tf
from test import test_single_img

def fix_img_dimension(img):
    height = img.shape[0]
    width = img.shape[1]
    height = height // 16 * 16
    width = width // 16 * 16
    return img[:height, :width],height,width

def nuset_func(img_gray):
    params = {}
    # Default values, most of them can be changed by user in the gui
    params['watershed'] = 'yes'
    params['min_score'] = 0.85
    params['nms_threshold'] = 0.1
    params['postProcess'] = 'yes'
    params['lr'] = 5e-5
    params['optimizer'] = 'rmsprop'
    params['epochs'] = 35
    params['normalization_method'] = 'fg'
    params['scale_ratio'] = 1.0
    params['model'] = 'NuSeT'

    if params['model'] == 'NuSeT':
        with tf.Graph().as_default():
            # Rescale the image if the nuclei is too small or too big

            img_use,h,w = fix_img_dimension(img_gray)
            img_mask = test_single_img(params, [img_use])
            return img_mask*255
            # Revert the image size to be the original one

            #cv2.imwrite(self.save_name, self.im_mask_np * 255)
#            im_mask = PIL.Image.fromarray((self.im_mask_np * 255))
