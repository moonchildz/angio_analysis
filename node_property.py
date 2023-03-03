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
def node_prop(node_list,th_img,br_img,cell_depth):
    #th_again = np.copy(th_img)

    r_list=[]
    n_list=[]
    e_list=[]
    d_list=[]
    cell_num =0
    draw = np.zeros_like(th_img)
    draw[th_img > 0.1] = 1.0
    draw[br_img > 0.1] = 0.1
    #print('NN',len(node))
    for node in node_list:
        black = 0
        r = 3
        #th_img[node[0],node[1]] = 0.5
        while(black==0 and r<1000):
            r+=1
            for angle in range(90):
                rad = (angle*4)*3.141592/180
                x = r*math.cos(rad)
                y = r*math.sin(rad)
                x_r = int(node[1]+x)
                y_r = int(node[0]+y)

                if x_r>=th_img.shape[1]:continue
                if y_r>=th_img.shape[0]:continue
                if x_r<0:continue
                if y_r<0:continue
                if th_img[y_r,x_r] <0.2:
                    black = 1
        #cv2.circle(th_again,(node[1],node[0]),int(r*1.2),(0),-1)
        r_list.append(r)

            #print(node[0],node[1],x_r, y_r,black)

        edge_num=0
        pre_x = -9999
        pre_y = -9999
        for angle in range(720):
            rad = (angle*0.5 ) * 3.141592 / 180
            x = r * math.cos(rad)
            y = r * math.sin(rad)
            x_r = int(node[1] + x)
            y_r = int(node[0] + y)
            if x_r ==pre_x and y_r == pre_y:continue
            if x_r >= br_img.shape[1]: continue
            if y_r >= br_img.shape[0]: continue
            if x_r < 0: continue
            if y_r < 0: continue
            if br_img[y_r, x_r] > 0.2:
                edge_num+=1
            pre_x = x_r
            pre_y = y_r
        e_list.append(edge_num)
        d_count = 0
        d_sum = 0
        for v in range(cell_depth.shape[0]):
            for u in range(cell_depth.shape[1]):
                if cell_depth[v,u] ==0: continue
                xx = u-node[1]
                yy = v-node[0]
                if xx*xx + yy*yy <r*r:
                    d_count+=1
                    d_sum+=cell_depth[v,u]
        d_sum/=(d_count+0.0001)
        if d_count ==0: d_sum=0
        n_list.append(d_count)
        d_list.append(d_sum)

        cv2.circle(draw,(node[1],node[0]),int(r/2),(d_sum*10),edge_num*2)
        print(r)
    cv2.imshow('Node_Stat', cv2.resize(draw,None,fx=0.5,fy=0.5))
    cv2.waitKey(20)

   # cv2.imwrite(dir_vessel+'props/'+vname[:-4]+'_nodestat.png',draw*255.0)
    #print('CD',r, d_count, d_sum, cell_depth[cell_depth>0])
    return r_list,e_list,n_list,d_list