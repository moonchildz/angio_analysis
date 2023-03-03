import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import ImageTk as itk
from PIL import Image
from tkinter import filedialog as fd
import NuSeT as nst
import img_process as ipr
import edge_property as eprop
import node_property as nprop
import os
import sys
import gc
import pandas as pd
from plantcv.plantcv import color_palette
from plantcv import plantcv as pcv
import datetime

window = tk.Tk()
class window_tk():
    def __init__(self,main):
        self.main=main
        self.canvas_2d = tk.Canvas(self.main, bg='white')
        self.canvas_2dTh = tk.Canvas(self.main, bg='white')

        self.btn_load = tk.Button(self.main,text = "Load Image",command = self.load_img)
        #self.btn_loadFolder = tk.Button(self.main,text = "Load Image Folder",command = self.load_imgFolder)
        self.btn_runImgProc = tk.Button(self.main,text = "Analyze image",command = self.runImgProc)
        self.btn_runStatistic = tk.Button(self.main,text = "Generate Graph",command = self.make_graph)
        self.progText = tk.Label(self.main, text='No file loaded')
        #self.progText.start(0)
        self.init_canvas(self.main)
        self.org_dir = None
        self.imgnum =0
        self.imgseq =0
        self.axis = None
        self.loaded = None
        self.imgList = []
        self.imgnameList = None
        self.length_of_category = []
        self.prop_of_all=[]
        self.name_of_all=[]
        self.category = []
        self.category_num = []
        self.category_now = None
        self.category_now_num = 0
        self.img_dpList=[]
        self.thval = 0
    def init_canvas(self,main):
        main.geometry('1327x712+100+50')
#        self.btn_loadFolder.pack(side=tk.BOTTOM)
        self.btn_load.pack(side=tk.BOTTOM)
        self.btn_runImgProc.pack(side=tk.BOTTOM)
        self.progText.pack(side=tk.BOTTOM)
        self.threshBar = Scale(main, from_=0, to=255, orient=HORIZONTAL,length=1024,command=self.onScaleChange)
        self.threshBar.pack(side=tk.BOTTOM)
        self.canvas_2d.config(width=650, height=350)
        self.canvas_2d.pack(side=tk.LEFT)
        self.canvas_2dTh.config(width=650, height=350)
        self.canvas_2dTh.pack(side=tk.RIGHT)
        self.img_route=None
        self.img=None
        self.img_dp=None
        self.dir_img=''
        today = datetime.date.today().strftime("%d%m_%Y_")
        now = datetime.datetime.now().strftime("%H%M%S")
        print(today,now)
        self.todayFilename = today+now+'_data.csv'
        self.img_dir = 'img_'+today+now+'/'
        if os.path.isdir(self.img_dir) is False :
            os.mkdir(self.img_dir)

    def load_img(self):
        gc.collect()
        ext_pic = {('JPG','*.jpg'),('TIF','*.tif'),('PNG',"*.png"),('BMP','*.bmp')}
        self.dir_img = fd.askopenfilename(parent=window,initialdir='C:/',title = "select a picture",
                                           filetypes=((ext_pic ) ))
        print(self.dir_img)
        self.img = cv2.imread(self.dir_img,1)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.img_dp = np.copy(self.img[:, :, 1])
        th = self.threshBar.get()
        print(th)
        self.display_img(self.img)
        self.loaded = 'File'
        self.progText.config(text='Loaded Image: '+self.dir_img)

    def load_imgFolder(self):
        gc.collect()
        self.dir_imgs = fd.askdirectory(parent=window,initialdir='C:/',title = "select a Directory ")
        self.imgnameList = os.listdir(self.dir_imgs)
        print(self.imgnameList)
        for il in self.imgnameList:
            print(il)
            if il.split('.')[-1] !='jpg' and il.split('.')[-1] !='png' and il.split('.')[-1] !='tif' and il.split('.')[-1] !='TIF' and il.split('.')[-1] !='bmp':
                print(il+'popped')
                self.imgnameList.pop(il)
        for il in self.imgnameList:
            img = cv2.imread(self.dir_imgs+'/'+il,1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.imgList.append(img)
        if len(self.imgList) ==0:return
        self.display_img(self.imgList[len(self.imgList)//2])
        self.category_now = self.dir_imgs.split('/')[-1]
        print(self.category_now)
        self.category.append(self.category_now)
        self.loaded = 'Folder'

    def display_img(self,img):
        imgitk = itk.PhotoImage(Image.fromarray(img))
        self.canvas_2d.create_image(0,0,anchor = tk.NW,image=imgitk)
        self.canvas_2d.image = imgitk
    def display_imgTh(self):
        imgitk = itk.PhotoImage(Image.fromarray(self.img_dp))
        self.canvas_2dTh.create_image(0,0,anchor = tk.NW,image=imgitk)
        self.canvas_2dTh.image = imgitk
    def onScaleChange(self,val):
        self.thval = int(val)
        if self.loaded =='File':
            self.img_dp = np.copy(self.img[:,:,0])
            print(val)
            self.img_dp[self.img_dp < int(val)] = 0
            self.img_dp[self.img_dp !=0] = 255
        elif self.loaded =='Folder':
            self.img_dp = np.copy(self.imgList[len(self.imgList)//2][:,:,0])
            print(val)
            self.img_dp[self.img_dp < int(val)] = 0
            self.img_dp[self.img_dp !=0] = 255

        self.display_imgTh()

    def runImgProc(self):
        self.progText.config(text='Hello World .')
        if self.loaded =='File':
            img_vesselTh = self.img_dp    # Threshold 완료된 혈관
            img_vessel = self.img[:,:,1]    # Threshold 완료된 혈관
            img_cell = self.img[:,:,2]  #cell 부분인 파랑 부분만 추출함
            img_cell_seg = nst.nuset_func(img_cell)
            img_refinedTh = self.filter_thImg(img_vesselTh)
            sk2, seg_br, sk_node, sk_edge = ipr.prepare_img(img_refinedTh)
#            length_list, distance_list = properties(seg_br, img_s, sk_node, sk_edge, s)
#            palette, cell_props,palette2 = ipr.cell_depth(img_cell,img_cell_seg)
            cell_points, cell_props, cell_paint = ipr.cell_depth(img_cell, img_cell_seg)
            all_properties = self.properties(seg_br, img_vesselTh, sk_node, sk_edge, img_cell, cell_points)
            self.prop_of_all.append(all_properties)
            self.progText.config(text='Done ! ')
            print('^^',all_properties)
            self.save_csv()
        if self.loaded =='Folder':
            self.img_dpList=[]
            if len(self.imgList) ==0:
                return
            th_val = int(self.threshBar.get())

            for i in self.imgList:
                img_Th = np.copy(i[:, :, 0])
                img_Th[img_Th < int(self.thval)] = 0
                img_Th[img_Th != 0] = 255
                img_vesselTh = img_Th  # Threshold 완료된 혈관
                img_vessel = i[:,:,1]
                img_cell = i[:,:,2]
                self.procSingleImg(img_vessel,img_vesselTh,img_cell)
                self.img_dpList.append(img_Th)
            self.length_of_category.append(len(self.imgList))
            self.category_now_num+=1
    def procSingleImg(self,img_vessel,img_vesselTh,img_cell):
        img_refinedTh = self.filter_thImg(img_vesselTh)
        sk2, seg_br, sk_node, sk_edge = ipr.prepare_img(img_refinedTh)
        img_cell_seg = nst.nuset_func(img_cell)
        img_refinedTh = self.filter_thImg(img_vesselTh)
        print('Skeleton is generated.')
        #            length_list, distance_list = properties(seg_br, img_s, sk_node, sk_edge, s)
        #            palette, cell_props,palette2 = ipr.cell_depth(img_cell,img_cell_seg)
        cell_points, cell_props, cell_paint = ipr.cell_depth(img_cell, img_cell_seg)
        print('Nucleus related parameters are derived.')
        all_properties = self.properties(seg_br, img_vesselTh, sk_node, sk_edge, img_cell, cell_points)
        print('Branch related Parameters are derived.')
        self.prop_of_all.append(all_properties)
        self.category_num.append(self.category_now_num)
        self.name_of_all.append(self.category_now)
        return

    def properties(self,branches, th_img, sk_node, sk_edge, cell_img, cell_pts):

        rand_color = color_palette(num=len(branches), saved=False)
        img_z = np.zeros_like(th_img)
        img_z2 = np.zeros_like(th_img)
        num_of_branch = len(branches)
        print('nob:', num_of_branch)
        large_count = 0
        euclidean_list = eprop.find_tips_and_euclidean(branches, sk_edge)
        len_list, width_list = eprop.find_width_and_arclength(branches, th_img)
        curvature_list = eprop.find_curvature(len_list, euclidean_list)
        print('Curvature Found')
        branch_pts = pcv.morphology.find_branch_pts(skel_img=sk_node)
        print('Branch Points Found')
        whole_len, whole_area, whole_peri = eprop.whole_len_area_peri(branches, th_img)
        print('Perimeter Found')
        peri, area = ipr.peri_area(th_img)
        print('Area Found')
        total_len = ipr.total_len(sk_edge)
        print('Branch All Length Found')
        node_list = []
        node_radius_list = []
        node_edgenum_list = []
        node_cellnum_list = []
        node_depth_list = []
        for v in range(branch_pts.shape[0]):
            for u in range(branch_pts.shape[1]):
                if branch_pts[v, u] > 0.1:
                    node_list.append([v, u])
        print('Node List All Appended :',len(node_list))
        nr_list, ne_list, nn_list, nd_list = nprop.node_prop(node_list, th_img, sk_node, cell_pts)  # r, edgenum, cellnum,depth 순서
        print('Node Properties Found')
        node_num = len(node_list)
        fill, fill_gray, label3 = pcv.morphology.fill_segments(th_img, branches)
        cv2.imwrite(self.img_dir+'fill_'+self.dir_img.split('/')[-1],fill)
        cv2.imshow('Fill', cv2.resize(fill, None, fx=0.5, fy=0.5))
        cv2.waitKey(20)
        # cv2.imwrite(dir_vessel+'props/'+vname[:-4]+'_fill.png',fill)
        fill_areas = np.zeros(len(label3))
        cell_num_per_areas = np.zeros(len(label3))
        mean_depth_per_areas = np.zeros(len(label3))
        for v in range(fill.shape[0]):
            for u in range(fill.shape[1]):
                if fill_gray[v, u] == 0: continue
                fill_areas[fill_gray[v, u] - 1] += 1
        for v in range(cell_pts.shape[0]):
            for u in range(cell_pts.shape[1]):
                if cell_pts[v, u] == 0: continue
                cell_num_per_areas[fill_gray[v, u] - 1] += 1
                mean_depth_per_areas[fill_gray[v, u] - 1] += cell_pts[v, u]
        mean_depth_per_areas /= cell_num_per_areas
        average_width = fill_areas / np.array(len_list)
        print('Now Saving property')

        w_row = []
        w_row_diff = []
        w_row_diff_mean = []

        for ww in width_list:
            w_row.append(ww)
            wwd = np.abs(np.diff(np.array(ww)))
            w_row_diff.append(ww)
            w_row_diff_mean.append(np.nanmean(wwd))
        f_row = [[len(label3)], curvature_list, fill_areas, len_list, euclidean_list,
                 mean_depth_per_areas, cell_num_per_areas, average_width,
                 [len(node_list)], nr_list, ne_list, nn_list, nd_list]
        f_row_mean = [
            np.nanmean(np.array(curvature_list)),  # Curvature
            np.nanmean(np.array(fill_areas)),  # Area
            np.nanmean(np.array(len_list)),  # Length
            np.nanmean(np.array(euclidean_list)),  # Euclidean Length
            np.nanmean(np.array(average_width)),  # Average Width
            np.nanmean(np.array(w_row_diff_mean)),  # Differentiate of Width
            whole_peri,  # Perimeter
            whole_area,  # Area
            len(label3),  # Number of Edge
            len(node_list),  # Number of Node
            np.nanmean(np.array(nr_list)),  # Node Radius
            np.nanmean(np.array(ne_list)),  # Num of Edge per Node
            np.nanmean(np.array(cell_num_per_areas)),  # Num of Cell in a Edge
            np.nanmean(np.array(nn_list)),  # Num of Cell in a Node
            np.nanmean(np.array(mean_depth_per_areas)),  # Depth of an Edge
            np.nanmean(np.array(nd_list))]  # Depth of a Node

        f_row_stddev = [
            np.nanstd(np.array(curvature_list)),  # Curvature
            np.nanstd(np.array(fill_areas)),  # Area
            np.nanstd(np.array(len_list)),  # Length
            np.nanstd(np.array(euclidean_list)),  # Euclidean Length
            np.nanstd(np.array(average_width)),  # Average Width
            np.nanstd(np.array(w_row_diff_mean)),  # Differentiate of Width
            np.nanstd(np.array(nr_list)),  # Node Radius
            np.nanstd(np.array(ne_list)),  # Num of Edge per Node
            np.nanstd(np.array(cell_num_per_areas)),  # Num of Cell in a Edge
            np.nanstd(np.array(nn_list)),  # Num of Cell in a Node
            np.nanstd(np.array(mean_depth_per_areas)),  # Depth of an Edge
            np.nanstd(np.array(nd_list))]  # Depth of a Node
        f_prop = []
        f_prop.extend(f_row_mean)
        f_prop.extend(f_row_stddev)
        #self.prop_of_all.append(f_prop)
        return f_prop
    def filter_thImg(self,img_th):
        img_gau = cv2.GaussianBlur(img_th, (5, 5), 0)
        img_gau = cv2.GaussianBlur(img_gau, (5, 5), 0)
        img_e = ipr.img_erode(img_gau,2)
        return img_e
    def save_csv(self):
        columns = [
            'Curvature_mean',
            'Area_mean',
            'Length_mean',
            'Euclid_Length_mean',
            'Width_mean',
            'Diff_Width_mean',

            'Whole_Perimeter',
            'Whole_Area',
            'Num_Edge',
            'Num_Node',

            'Node_Radius_mean',
            'Num_Edge_per_Node_mean',
            'Num_Cell_per_Edge_mean',
            'Num_Cell_per_Node_mean',
            'Depth_Edge_mean',
            'Depth_Node_mean',

            'Curvature_std',
            'Area_std',
            'Length_std',
            'Euclid_Length_std',
            'Width_std',
            'Diff_Width_std',
            'Node_Radius_std',
            'Num_Edge_per_Node_std',
            'Num_Cell_per_Edge_std',
            'Num_Cell_per_Node_std',
            'Depth_Edge_std',
            'Depth_Node_std',
                   ]

        prop_whole_copy = np.array(self.prop_of_all)
        self.todayDF = pd.DataFrame(prop_whole_copy,columns=columns)
        self.todayDF.to_csv(self.todayFilename,index=False)
        print('csv Written')

        return
    def make_graph(self):
        poa_np = np.array(self.prop_of_all)
        num_data = poa_np.shape[0]
        if num_data==0:return

app= window_tk(window)
window.mainloop()