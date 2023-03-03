import cv2
import numpy as np
import os
import re
import sys
from scipy.spatial import Delaunay,Voronoi,SphericalVoronoi
from skimage.morphology import skeletonize,thin
from skimage import data
from skimage.util import invert
from plantcv import plantcv as pcv
import math
dir_vessel = 'D:\Vessel_210310\Vasculogenesis (Radiation)/'
dir_vessel = 'D:\Vessel_210310\Angiogenesis (VEGF-A)/'
dir_vessel = 'D:/download/vascular_brighter_seg\EXP2/'
#dir_vessel = 'D:/download/vascular_brighter_seg\EXP2/'
dir_vessel_name = 'D:/download/vascular_brighter\EXP2/'

#dir_vessel = 'D:\Vessel_210310\Angiogenesis (VEGF-A + S1P)/'
def edge_thickness(edge,br_img):
    length_list=[]
    for en,ep in enumerate(edge):
        if en>len(edge)-5:continue
        ex = ep[0][0]
        ey = ep[0][1]
        ep_ = edge[en+4]
        ex_ = ep_[0][0]
        ey_ = ep_[0][1]
        def max_abs(value):
            if value>=0:
                return max(value,0.000001)
            else:
                return min(value,-0.000001)
        dxdy = ( max_abs(ey_-ey) /(max_abs(ex_-ex)))
        norm = -1/dxdy
        #print(dxdy,'::::',norm)
        found =0
        xplus = 1
        while(found ==0):
            if abs(dxdy)>1:
                x_right = ex+xplus
                y_right = int(norm*xplus+ey)
            else:
                y_right = ey + xplus
                x_right = int(dxdy * xplus + ex)
            #print(y_right,x_right)
            if x_right>=br_img.shape[1]:break
            if x_right<0:break
            if y_right>=br_img.shape[0]:break
            if y_right<0:break
            br_px_r = br_img[y_right,x_right]

            if br_px_r <0.2:
                found = 1
                break
            xplus+=1
        length_right = math.sqrt(xplus*xplus+(y_right-ey)*(y_right-ey))

        xminus = 1
        found=0
        while(found ==0):
            if abs(dxdy)>1:
                x_left = ex-xminus
                y_left = int(ey-norm*xminus)
            else:
                y_left = ey + xminus
                x_left = int(-dxdy * xplus + ex)
            if x_left>=br_img.shape[1]:break
            if x_left<0:break
            if y_left>=br_img.shape[0]:break
            if y_left<0:break

            br_px_l = br_img[y_left,x_left]
            if br_px_l <0.2:
                found = 1
                break
            xminus+=1
        length_left = math.sqrt(xplus*xplus+(y_right-ey)*(y_right-ey))

        length_list.append(length_left + length_right)
   # print('length of each',en,':' ,length_list)
    return np.array(length_list)
def find_tips_and_euclidean(edge,br_img):
    edge_test = np.zeros_like(br_img)
    endpoint1 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [-1, 1, -1]])
    endpoint2 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, 1]])

    endpoint3 = np.rot90(endpoint1)
    endpoint4 = np.rot90(endpoint2)
    endpoint5 = np.rot90(endpoint3)
    endpoint6 = np.rot90(endpoint4)
    endpoint7 = np.rot90(endpoint5)
    endpoint8 = np.rot90(endpoint6)

    endpoints = [endpoint1, endpoint2, endpoint3, endpoint4, endpoint5, endpoint6, endpoint7, endpoint8]
    tipsnum=0
    euclidean =[]
    for e in edge:
        tips = []
        for i,ee in enumerate(e):
            e_x = ee[0][0]
            e_y = ee[0][1]
            if e_x<2: e_x = 2
            if e_x>br_img.shape[1]-2: e_x = br_img.shape[1]-2
            if e_y<2: e_y = 2
            if e_y>br_img.shape[0]-2: e_y = br_img.shape[0]-2

            br_part = br_img[e_y-1:e_y+2,e_x-1:e_x+2]//254
            if br_part.shape!=(3,3):
                print(br_part)
                print('Err',e_x,e_y)
                #cv2.imshow('br', br_img)
                #cv2.waitKey()
            for ep in endpoints:
                deci = ep * br_part
                sum = np.sum(deci)
                if sum ==2:
                    tipsnum+=1
                    tips.append(ee[0])
        if len(tips)!=2:
            euclidean.append(-1.0)
        else:
            xd = tips[0][0]-tips[1][0]
            yd = tips[0][1]-tips[1][1]
            length = math.sqrt(xd*xd+yd*yd)
            euclidean.append(length)
        tipsnum=0
        #    print(br_part)
    return euclidean
def find_width_and_arclength(branches,th_img):
    len_list=[]
    width_list=[]
    for br in branches:
        width = edge_thickness(br,th_img)
        width_list.append(width)
        length = cv2.arcLength(br,False)
        len_list.append(length)
    return len_list,width_list
def find_curvature(len_list,eu_list):
    print('list_length:',len(len_list),len(eu_list))
    c_list=[]
    for l,e in zip(len_list,eu_list):
        if e>0:
            c_list.append(l/e)
        else:
            c_list.append(1.0)
    return c_list
def whole_len_area_peri(branches,th_img):
    whole_length = 0
    perimeter = 0
    area = 0
    for br in branches:
        length = cv2.arcLength(br,False)
        whole_length+=length
    contours, hier = cv2.findContours((th_img * 255.0).astype(np.uint8), mode=cv2.RETR_TREE,
                                      method=cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        clen = cv2.arcLength(c,False)
        perimeter+=clen
    for v in range(th_img.shape[0]):
        for u in range(th_img.shape[1]):
            if th_img[v,u] >0.5:
                area+=1
    return whole_length,area,perimeter
