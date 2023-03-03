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
from matplotlib import pyplot as plt
import math
dir_vessel = 'D:\Vessel_210310\Vasculogenesis (Radiation)/'
dir_vessel = 'D:\Vessel_210310\Angiogenesis (VEGF-A)/'
dir_vessel = 'D:/download/vascular_brighter_seg\EXP2/'
dir_vessel_name = 'D:/download/vascular_brighter\EXP2/'
#dir_vessel = 'D:/download/vascular_brighter_seg\EXP2/'

#dir_vessel = 'D:\Vessel_210310\Angiogenesis (VEGF-A + S1P)/'
h_start = 200
h_end = 1000
def load_images(ibody,icell,iseg):
    #print(filename)
    img_cell = cv2.imread(icell,-1)[40:,:]/255.0
    img_seg = cv2.imread(iseg,-1)[40:,:]/255.0
    img_body = (cv2.imread(ibody,-1)[40:,:,2])
    print(np.max(img_body),np.max(img_cell),np.max(img_seg))

    return img_body/np.max(img_body),img_cell,img_seg
def cell_depth(cell_img,cell_seg):
    cell_sx = cv2.Sobel(cell_img,cv2.CV_64F,1,0,ksize=3)
    cell_sy = cv2.Sobel(cell_img,cv2.CV_64F,0,1,ksize=3)
    cell_sobel = 0.5*np.abs(cell_sx)+0.5*np.abs(cell_sy)
    #print(np.max(cell_sobel))
    cell_eroded = img_erode(cell_seg,3)
    palette = np.zeros_like(cell_img)
    palette2 = np.zeros_like(cell_img)
    contours, hier = cv2.findContours((cell_eroded*255.0).astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(palette,contours,-1,(255),2)
    cell_props=[]
    print(palette.shape)
    for c in contours:
        if len(c)<6:continue
        e = cv2.fitEllipse(c)
        #print(e)
        pseudo_depth = find_sharpness_xy(cell_sobel,int(e[0][0]),int(e[0][1]),int(e[1][0]//2+6),int(e[1][1]//2+6))
        cell_props.append(
            (e[0][0],e[0][1],pseudo_depth,e)
        )
        if e[0][0]>palette.shape[1] or e[0][0]<0:continue
        if e[0][1]>palette.shape[0] or e[0][1]<0:continue
        palette[int(e[0][1]),int(e[0][0])] = pseudo_depth
        cv2.ellipse(palette2, e, 200 , -1)
    return palette,cell_props,palette2
def img_erode(img,iter):
    k = np.ones((3, 3), np.uint8)
    k[0,0] = 0
    k[0,2]=0
    k[2,0]=0
    k[2,2]=0

    eroded = cv2.erode(img,k,iterations=iter)
    return eroded
def img_dilate(img,iter):
    k = np.ones((3, 3), np.uint8)
    k[0,0] = 0
    k[0,2]=0
    k[2,0]=0
    k[2,2]=0

    dilated = cv2.dilate(img,k,iterations=iter)
    return dilated
def find_sharpness_xy(img_sharp,x,y,k_y,k_x):
    img_patch = img_sharp[y-k_y:y+k_y,x-k_x:x+k_x]
    img_patch[img_patch <0.01] = np.nan
#    print(x,y,k_y,k_x,img_patch.shape)
    sharpness = np.nanmean(img_patch)
    if np.isnan(sharpness) : sharpness =0
    #print(sharpness1,sharpness,sharpness2)
    return sharpness
def prepare_img(img_th):
#
    sk_node = skeletonize(img_th, method='lee')
    pr, seg, eo = pcv.morphology.prune(skel_img=sk_node, size=20)
    sk_edge = np.zeros_like(pr)
    eo_=[]
    for e in eo:
        if len(e)<5:continue
        eo_.append(e)
        for ee in e:
            sk_edge[ee[0][1],ee[0][0]]=255

    print('pruned length: ',len(eo),len(eo_))
    #cv2.imshow('Prune', pr)

#print('whats this',pr.shape,pr.dtype)
    return pr,eo_,sk_node,sk_edge
def peri_area(img):
    area = 0
    perimeter = 0
    for v in range(img.shape[0]):
        for u in range(img.shape[1]):
            if img[v,u]>0.5:
                area+=1
    contours, hier = cv2.findContours(img.astype(np.uint8), mode=cv2.RETR_TREE,
                                      method=cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        clen = cv2.arcLength(c,False)
        perimeter+=clen
    return area,perimeter
def total_len(img):
    tl = 0
    for v in range(img.shape[0]):
        for u in range(img.shape[1]):
            if img[v,u]>0.5:
                tl+=1
    return tl
def prepare_img2(img,img_cell,img_seg):
    cv2.imshow('original',cv2.resize(img,None,fx=0.5,fy=0.5))
    cv2.waitKey(20)
    cell_points, cell_props, cell_paint = cell_depth(img_cell, img_seg)
    img_draw = np.zeros((img.shape[0], img.shape[1]))
    # img_rg = img[:,:,2]+img[:,:,1]
    img_gau = cv2.GaussianBlur(img, (5, 5), 0)
    img_gau = cv2.GaussianBlur(img_gau, (5, 5), 0)
    img_med = np.median(img_gau)
    img_mean = np.mean(img_gau)
    print(np.max(img),np.min(img),np.mean(img),np.nanmean(img))
    dummy, th = cv2.threshold(img_gau, img_mean, 1.0, cv2.THRESH_BINARY)
#    cv2.imshow('th1',cv2.resize(th,None,fx=0.5,fy=0.5))

    img_e = img_erode(th, 1)
#    cv2.imshow('th2',cv2.resize(img_e,None,fx=0.5,fy=0.5))

    i_c, hier = cv2.findContours((img_e).astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    for cs in i_c:  ## 빈칸 erode dilate 동시에 해서 망가진 부분 없애기
        if len(cs) < 60 :
            img_temp = np.zeros((img.shape[0], img.shape[1]))
            cv2.drawContours(img_temp, [cs], -1, (1), -1)
            img_temp_dil = img_dilate(img_temp,1)
            i_c2, hier = cv2.findContours((img_temp_dil).astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

            cs_mean=0
            cs_count=0
            for cs2 in i_c2:
                for css2 in cs2:
                    cs_mean+=th[css2[0][1],css2[0][0]]
                    cs_count+=1
            cs_mean /=(cs_count+0.00001)
            #print(cs_mean,'mean')
            if cs_mean / cs_count > 0.5:
                cv2.fillPoly(img_e, pts=[cs], color=(0))
            else:
                cv2.fillPoly(img_e, pts=[cs], color=(1))

    i_c_, hier = cv2.findContours((img_e).astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    i_c__ =[]
    for i,c in enumerate(i_c_):
        if len(c) >=7 :
            i_c__.append(c)
        continue
    img_draw = np.zeros((img.shape[0], img.shape[1]))
    img_final = np.zeros((img.shape[0], img.shape[1]))
    img_test = np.zeros((img.shape[0], img.shape[1]))
    # Simplify very large segments
    cv2.drawContours(img_test, i_c, -1, (1), 1)

    for cs in i_c__:

        peri = cv2.arcLength(cs, True)
        if peri < 30 :continue
        print(peri)
        img_temp = np.zeros((img.shape[0], img.shape[1]))
        for i in range(10):
            cs = cv2.approxPolyDP(cs, (i+1)*8/peri, True)
        cv2.drawContours(img_temp, [cs], -1, (1), -1)
        img_ero = img_erode(img_temp,5)
        ic2, hier = cv2.findContours((img_ero*255.0).astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(img_test, ic2, -1, (0.8), 2)


        for c2 in ic2:
            mean_c = 0
            count_c = 0
            for c3 in c2:
                count_c+=1
                mean_c+=img_e[c3[0][1],c3[0][0]]
            mean_c/=(count_c+0.00001)
            #print('mean',mean_c)
            if mean_c>10:continue
            if mean_c>0.5:
                cv2.drawContours(img_final, [cs], -1, (1.0), -1)
                #cv2.drawContours(img_test, [cs], -1, (1.0), 1)

    for cs in i_c__:

        peri = cv2.arcLength(cs, True)
        if peri < 30 :continue

        img_temp = np.zeros((img.shape[0], img.shape[1]))
        for i in range(10):
            cs = cv2.approxPolyDP(cs, (i+1)*8/peri, True)
        cv2.drawContours(img_temp, [cs], -1, (1), -1)
        img_ero = img_erode(img_temp,3)
        ic2, hier = cv2.findContours((img_ero*255.0).astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
       # cv2.drawContours(img_test, ic2, -1, (0.6), 1)


        for c2 in ic2:
            mean_c = 0
            count_c = 0
            for c3 in c2:
                count_c+=1
                mean_c+=img_e[c3[0][1],c3[0][0]]
            mean_c/=(count_c+0.00001)
            #print('mean',mean_c)
            if mean_c>10:continue
            if mean_c<0.5:
                cv2.drawContours(img_final, [cs], -1, (0), -1)
      #          cv2.drawContours(img_test, [cs], -1, (1.0), 0)



    sk_node = skeletonize(img_final, method='lee')
    pr, seg, eo = pcv.morphology.prune(skel_img=sk_node, size=20)
    sk_edge = np.zeros_like(pr)
    eo_=[]
    for e in eo:
        if len(e)<5:continue
        eo_.append(e)
        for ee in e:
            sk_edge[ee[0][1],ee[0][0]]=255

  #  print('pruned length: ',len(eo),len(eo_))
  #  cv2.imshow('sk_node', cv2.resize(sk_node,None,fx=0.5,fy=0.5))
    cv2.imshow('th_final', cv2.resize(img_final,None,fx=0.5,fy=0.5))
    cv2.waitKey(20)

  #  cv2.imshow('th_test', cv2.resize(img_test,None,fx=0.5,fy=0.5))
  #  cv2.waitKey()
    return img_e,img_final, pr,eo_,sk_node,sk_edge
