import argparse
import os
import shutil
from os.path import splitext
import h5py
#import nibabel as nib
import numpy as np
# import SimpleITK as sitk
import torch
#from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from os import listdir
# from networks.efficientunet import UNet
from networks.net_factory import net_factory
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,roc_curve,recall_score,auc,precision_recall_curve,average_precision_score,f1_score
from glob import glob
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2300000000
import time
from pyheatmap.heatmap import HeatMap
CELL_X = 256
CELL_Y = 256
row_cell = 5120
col_cell = 5120
NetName = 'DeepLab_RGBD'
CLASS_NUM = 7
def tans_img_rgb(img_A, ds_array_B, img_newB):
    img_tif = Image.open(img_A)
    width = img_tif.size[0]
    height = img_tif.size[1]
    img = np.asarray(img_tif).copy()
    # img = np.zeros([height, width , 3], dtype=np.uint8)
    ds_array_B_r = img[:,:,0]
    ds_array_B_g = img[:,:,1]
    ds_array_B_b = img[:,:,2]

    # ds_array_B_r[ds_array_B == 0] = 0
    # ds_array_B_g[ds_array_B == 0] = 127
    # ds_array_B_b[ds_array_B == 0] = 204
    #
    # ds_array_B_r[ds_array_B == 1] = 123
    # ds_array_B_g[ds_array_B == 1] = 59
    # ds_array_B_b[ds_array_B == 1] = 24
    #
    # ds_array_B_r[ds_array_B == 2] = 255
    # ds_array_B_g[ds_array_B == 2] = 192
    # ds_array_B_b[ds_array_B == 2] = 0
    #
    # ds_array_B_r[ds_array_B == 3] = 217 #255,255,0粉砂
    # ds_array_B_g[ds_array_B == 3] = 217
    # ds_array_B_b[ds_array_B == 3] = 217
    #
    # ds_array_B_r[ds_array_B == 4] = 56
    # ds_array_B_g[ds_array_B == 4] = 168
    # ds_array_B_b[ds_array_B == 4] = 0
    ds_array_B_r[ds_array_B == 3] = 180
    ds_array_B_g[ds_array_B == 3] = 180
    ds_array_B_b[ds_array_B == 3] = 180  # 泥岩

    ds_array_B_r[ds_array_B == 1] = 255
    ds_array_B_g[ds_array_B == 1] = 255
    ds_array_B_b[ds_array_B == 1] = 128  # 砂岩

    ds_array_B_r[ds_array_B == 2] = 240
    ds_array_B_g[ds_array_B == 2] = 116
    ds_array_B_b[ds_array_B == 2] = 0  # 砾岩

    ds_array_B_r[ds_array_B == 5] = 0
    ds_array_B_g[ds_array_B == 5] = 255
    ds_array_B_b[ds_array_B == 5] = 0  # 植被1

    ds_array_B_r[ds_array_B == 6] = 60
    ds_array_B_g[ds_array_B == 6] = 255
    ds_array_B_b[ds_array_B == 6] = 0  # 植被2

    ds_array_B_r[ds_array_B == 4] = 100
    ds_array_B_g[ds_array_B == 4] = 100
    ds_array_B_b[ds_array_B == 4] = 100  # 浮土

    img[:, :, 0] = ds_array_B_r
    img[:, :, 1] = ds_array_B_g
    img[:, :, 2] = ds_array_B_b
    # 写入影像数据
    im = Image.fromarray(img)
    im = im.convert("RGB")
    im.save(img_newB)
    del img


def tans_img_ys(img_A, ds_array_B, img_newB):
    datasetA = gdal.Open(img_A)  # 打开文件，用这个tif图的投影信息
    datasetA_geotrans = datasetA.GetGeoTransform()  # 仿射矩阵

    datasetA_proj = datasetA.GetProjection()  # 地图投影信息

    im_width = datasetA.RasterXSize  # 栅格矩阵的列数
    im_height = datasetA.RasterYSize  # 栅格矩阵的行数
    # im_bands = datasetA.RasterCount
    im_bands = 1
    # datatype = gdal.GDT_Byte
    # list2 = [gdal.GDT_Byte,gdal.GDT_Byte,gdal.GDT_UInt16,gdal.GDT_Int16,gdal.GDT_UInt32,gdal.GDT_Int32,gdal.GDT_Float32,gdal.GDT_Float64,gdal.GDT_CInt16,gdal.GDT_CInt32,gdal.GDT_CFloat32,gdal.GDT_CFloat64]
    datatype = gdal.GDT_Byte
    # datatype = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
    dataset = driver.Create(img_newB, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(datasetA_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(datasetA_proj)  # 写入投影1
    # dataset.GetRasterBand(1).WriteArray(ds_array_B)
    # 写入影像数据


    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(ds_array_B)
    del dataset


def tans_single_img(img_A, ds_array_B, img_newB):
    datasetA = gdal.Open(img_A)  # 打开文件，用这个tif图的投影信息
    datasetA_geotrans = datasetA.GetGeoTransform()  # 仿射矩阵

    datasetA_proj = datasetA.GetProjection()  # 地图投影信息

    im_width = datasetA.RasterXSize  # 栅格矩阵的列数
    im_height = datasetA.RasterYSize  # 栅格矩阵的行数
    # im_bands = datasetA.RasterCount
    im_bands = 3
    # datatype = gdal.GDT_Byte
    # list2 = [gdal.GDT_Byte,gdal.GDT_Byte,gdal.GDT_UInt16,gdal.GDT_Int16,gdal.GDT_UInt32,gdal.GDT_Int32,gdal.GDT_Float32,gdal.GDT_Float64,gdal.GDT_CInt16,gdal.GDT_CInt32,gdal.GDT_CFloat32,gdal.GDT_CFloat64]
    datatype = gdal.GDT_Byte
    # datatype = gdal.GDT_Float32
    driver = gdal.GetDriverByName("GTiff")  # 创建文件驱动
    dataset = driver.Create(img_newB, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(datasetA_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(datasetA_proj)  # 写入投影1
    # dataset.GetRasterBand(1).WriteArray(ds_array_B)
    # 写入影像数据
    img_tif = Image.open(img_A)

    img = np.asarray(img_tif).copy()

    # img = np.zeros([im_height, im_width, 3], dtype=np.uint8)
    ds_array_B_r = img[:, :, 0]
    ds_array_B_g = img[:, :, 1]
    ds_array_B_b = img[:, :, 2]
    # ds_array_B_r = img[0,:, :]
    # ds_array_B_g = img[1,:, :]
    # ds_array_B_b = img[2,:, :]
    ds_array_B_r[ds_array_B == 3] = 180
    ds_array_B_g[ds_array_B == 3] = 180
    ds_array_B_b[ds_array_B == 3] = 180  # 泥岩

    ds_array_B_r[ds_array_B == 1] = 255
    ds_array_B_g[ds_array_B == 1] = 255
    ds_array_B_b[ds_array_B == 1] = 128  # 砂岩

    ds_array_B_r[ds_array_B == 2] = 240
    ds_array_B_g[ds_array_B == 2] = 116
    ds_array_B_b[ds_array_B == 2] = 0  # 砾岩

    ds_array_B_r[ds_array_B == 5] = 0
    ds_array_B_g[ds_array_B == 5] = 255
    ds_array_B_b[ds_array_B == 5] = 0  # 植被1

    ds_array_B_r[ds_array_B == 6] = 60
    ds_array_B_g[ds_array_B == 6] = 255
    ds_array_B_b[ds_array_B == 6] = 0  # 植被2

    ds_array_B_r[ds_array_B == 4] = 100
    ds_array_B_g[ds_array_B == 4] = 100
    ds_array_B_b[ds_array_B == 4] = 100  # 浮土



    # ds_array_B_r[ds_array_B == 4] = 0  # 255,255,0粉砂
    # ds_array_B_g[ds_array_B == 4] = 255
    # ds_array_B_b[ds_array_B == 4] = 0
    img[:, :, 0] = ds_array_B_r
    img[:, :, 1] = ds_array_B_g
    img[:, :, 2] = ds_array_B_b
    dataset.GetRasterBand(1).WriteArray(ds_array_B_r)
    dataset.GetRasterBand(2).WriteArray(ds_array_B_g)
    dataset.GetRasterBand(3).WriteArray(ds_array_B_b)
    # for i in range(im_bands):
    #     dataset.GetRasterBand(i+1).WriteArray(img[i])
    del dataset
def tans_img(img_ij,mask):


    img = img_ij.copy()

    # img = np.zeros([img_ij.shape[0], img_ij.shape[1], 3], dtype=np.uint8)
    ds_array_B_r = img[:, :, 0]
    ds_array_B_g = img[:, :, 1]
    ds_array_B_b = img[:, :, 2]


    # ds_array_B_r = img[0,:, :]
    # ds_array_B_g = img[1,:, :]
    # ds_array_B_b = img[2,:, :]
    ds_array_B_r[mask == 1] = 255
    ds_array_B_g[mask == 1] = 255
    ds_array_B_b[mask == 1] = 128  # 砂岩

    ds_array_B_r[mask == 2] = 240
    ds_array_B_g[mask == 2] = 116
    ds_array_B_b[mask == 2] = 0  # 砾岩

    ds_array_B_r[mask == 3] = 180
    ds_array_B_g[mask == 3] = 180
    ds_array_B_b[mask == 3] = 180  # 泥岩

    ds_array_B_r[mask == 4] = 100
    ds_array_B_g[mask == 4] = 100
    ds_array_B_b[mask == 4] = 100  # 浮土

    ds_array_B_r[mask == 5] = 0
    ds_array_B_g[mask == 5] = 255
    ds_array_B_b[mask == 5] = 0  # 植被1

    ds_array_B_r[mask == 6] = 60
    ds_array_B_g[mask == 6] = 255
    ds_array_B_b[mask == 6] = 0  # 植被2
    # ds_array_B_r[ds_array_B == 4] = 0  # 255,255,0粉砂
    # ds_array_B_g[ds_array_B == 4] = 255
    # ds_array_B_b[ds_array_B == 4] = 0
    img[:, :, 0] = ds_array_B_r
    img[:, :, 1] = ds_array_B_g
    img[:, :, 2] = ds_array_B_b
    return img






def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(r'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(r'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
def predict_img_old(net,
                full_img,
                scale_factor=1,
                out_threshold=0.5):
    x, y = full_img.shape[0], full_img.shape[1]
    slice = zoom(full_img, (256 / x, 256 / y,1), order=0)
    slice = np.transpose(slice, (2, 0, 1))
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0)
    return pred
def Nom01(img):
    min = np.min(img)
    max = np.max(img)+0.000001
    imgNew = (img-min)/(max-min)
    # img = imgNew.astype('uint8')
    return imgNew
def RGB01(img):
    img = img.astype('int16')
    r = img[:,:,0:1]
    g = img[:, :, 1:2]
    b = img[:, :, 2:3]
    h = img[:, :, 3:4]
    r01 = r.copy()
    g01 = g.copy()
    b01 = b.copy()
    h01 = h.copy()
    rgb = r+ g +b
    r01 = r01/(rgb+0.000001)
    g01 = g01/(rgb+0.000001)
    b01 = b01 / (rgb+0.000001)
    img_rgb01 = np.concatenate((r01,g01,b01,h01),axis = 2)
    return img_rgb01
def BGR01(img):
    img = img.astype('int16')
    r = img[:,:,0:1]
    g = img[:, :, 1:2]
    b = img[:, :, 2:3]
    h = img[:, :, 3:4]
    r01 = r.copy()
    g01 = g.copy()
    b01 = b.copy()
    h01 = h.copy()
    rgb = r+ g +b
    r01 = r01/(rgb+0.000001)
    g01 = g01/(rgb+0.000001)
    b01 = b01 / (rgb+0.000001)
    img_rgb01 = np.concatenate((b01,g01,r01,h01),axis = 2)#注意CV2顺序是BGR,需要转换下
    return img_rgb01

def rgb(img):
    r = img[:, :, 0:1]
    g = img[:, :, 1:2]
    b = img[:, :, 2:3]
    h = img[:, :, 3:4]
    r01 = r.copy()
    g01 = g.copy()
    b01 = b.copy()
    h01 = h.copy()
    img_rgb01 = np.concatenate((b01, g01, r01, h01), axis=2)  # 注意CV2顺序是BGR,需要转换下
    return img_rgb01

def Nom255(img):
    min = np.min(img)
    max = np.max(img) + 1e-6
    imgNew = img*255/(max-min)
    img = imgNew.astype('uint8')
    return img
def predict_img(net,
                full_img,
                scale_factor=1,
                out_threshold=0.5):
    x, y = full_img.shape[0], full_img.shape[1]
    slice = zoom(full_img, (256 / x, 256 / y,1), order=0)
    slice = np.transpose(slice, (2, 0, 1))
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out_main = net(input)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0)
    return pred
def  result_Normalization(result):
    array = result.cpu().data.numpy()

    return array
def writeTiff(image, fileName,geotrans,proj):
    bandCount = image.shape[2]
    col = image.shape[1]
    row = image.shape[0]
    driver = gdal.GetDriverByName("GTiff")
    dataset_result = driver.Create(fileName, col, row, bandCount, gdal.GDT_Byte)
    dataset_result.SetGeoTransform(geotrans)  # 写入仿射变换参数
    dataset_result.SetProjection(proj)  # 写入投影1
    for i in range(bandCount):
      dataset_result.GetRasterBand(i+1).WriteArray(image[:,:,i])
    del dataset_result

def identify_datu(image_name,dsm_name,model_path):

    dataset_img = gdal.Open(image_name)
    dataset_dsm = gdal.Open(dsm_name)
    head = dataset_img.GetGeoTransform()
    geotrans = dataset_img.GetGeoTransform()  # 仿射矩阵
    proj = dataset_img.GetProjection()  # 地图投影信息
    col = dataset_img.RasterXSize  # 栅格矩阵的列数shape2
    row = dataset_img.RasterYSize  # 栅格矩阵的行数shape1
    output = os.path.dirname(image_name) + '/result/' + os.path.basename(image_name).split('.')[0] + '.TIF'
    if col > col_cell or row > row_cell:
        colnum = int(col / col_cell) + 1
        rownum = int(row / row_cell) + 1
        result = np.zeros([row,col, 3], dtype=np.uint8)
        for r in range(rownum):
            for c in range(colnum):
                row_i = row_cell
                col_i = col_cell
                c_min = c * row_cell
                r_min = r * col_cell
                if c_min > 0:
                    c_min = c_min -128
                if r_min > 0:
                    r_min = r_min - 128
                ci_max = c_min + col_cell
                ri_max = r_min + row_cell
                if ci_max > col:
                    col_i = col - c_min
                if ri_max > row:
                    row_i = row - r_min
                if c_min < col and r_min < row:
                    img_ij = dataset_img.ReadAsArray(c_min, r_min, col_i, row_i)
                    dsm_ij = dataset_dsm.ReadAsArray(c_min, r_min, col_i, row_i)
                    result_ij = identify_img(img_ij, dsm_ij, model_path, CELL_X, CELL_Y)
                    ci = ci_max-c_min
                    ri = ri_max-r_min
                    result[r_min:ri_max,c_min:ci_max,:] = result_ij
        print(result.shape)
        writeTiff(result, output,geotrans,proj)
    else :
        identify_single_img(image_name, dsm_name, model_path, CELL_X, CELL_Y)


def identify_single_img(image_name, dsm_name, model_path, row, col):
    image = cv2.imread(image_name)
    # image = RGB01(image)
    hsm = cv2.imread(dsm_name, -1)
    hsm = Nom01(hsm)
    hsm = hsm.reshape((hsm.shape[0], hsm.shape[1], 1))
    image = np.concatenate((image, hsm), axis=2)
    rowheight = int(row)
    colwidth = int(col)
    h_img = int(image.shape[0])
    w_img = int(image.shape[1])
    buchang = 0.5
    rownum = int(h_img / (rowheight * buchang)) + 1
    colnum = int(w_img / (colwidth * buchang)) + 1
    # net = net_factory(net_type='unet', in_chns=3,
    #                  class_num=4)
    net = net_factory(net_type=NetName,
                      class_num=CLASS_NUM)
    net.load_state_dict(torch.load(model_path))
    mask = np.zeros((h_img, w_img))
    for r in range(rownum):
        for c in range(colnum):
            strXmin = float(c * colwidth * buchang)
            strYmin = float(r * rowheight * buchang)
            r1 = r * rowheight * buchang
            r2 = (r * buchang + 1) * rowheight
            if r2 > h_img:
                r2 = h_img - 1
            c1 = c * colwidth * buchang
            c2 = (c * buchang + 1) * colwidth
            if c2 > w_img:
                c2 = w_img - 1
            c2 = (c * buchang + 1) * colwidth
            c1 = int(c1)
            c2 = int(c2)
            r1 = int(r1)
            r2 = int(r2)
            # print(c1,c2,r1,r2)
            if r1 >= (h_img - 1) or c1 >= (w_img - 1):
                continue
            img = image[r1:r2, c1:c2]
            img = RGB01(img)
            # image[r1:r2, c1:c2]
            # img=img/255

            maski = predict_img(net, img)


            if r1 > 0:
                r1 = int(r1 + rowheight / 4)

            if c1 > 0:
                c1 = int(c1 + colwidth / 4)

            if r1 == 0:
                ri1 = 0
            else:
                ri1 = int(rowheight / 4)
            if c1 == 0:
                ci1 = 0
            else:
                ci1 = int(colwidth / 4)

            ri2 = int(0.75 * rowheight)
            ci2 = int(0.75 * colwidth)
            if c2 > (w_img):
                c2 = (w_img)
            if r2 > (h_img):
                r2 = (h_img)
            if c2 <= (w_img - colwidth / 4):
                c2 = int(c2 - colwidth / 4)
            else:
                ci2 = int(ci1 + c2 - c1)
            if r2 <= (h_img - rowheight / 4):
                r2 = int(r2 - rowheight / 4)
            else:
                ri2 = int(ri1 + r2 - r1)
            mask[r1:r2, c1:c2] = maski[ri1:ri2, ci1:ci2]
    output = os.path.dirname(image_name) + '/result/' + os.path.basename(image_name).split('.')[0] + '.TIF'
    tans_single_img(image_name, mask, output)

###多波段##################
def identify_img(image,dsm,model_path,row,col):
     image = np.transpose(image, (1, 2, 0))
     # dsm = np.transpose(dsm, (1, 2, 0))
     # image01 = image / 255.0
     image01  = image
     dsm = Nom01(dsm)
     dsm = dsm.reshape((dsm.shape[0], dsm.shape[1], 1))
     image_hb = np.concatenate((image01, dsm), axis=2)
     # label = cv2.imread(label_name,cv2.IMREAD_GRAYSCALE)
     rowheight = int(row)
     colwidth = int(col)
     h_img = int(image.shape[0])
     w_img = int(image.shape[1])
     buchang=0.5
     rownum=int(h_img/(rowheight*buchang)) +1
     colnum=int(w_img/(colwidth*buchang)) +1
     # net = net_factory(net_type='unet', in_chns=3,
     #                  class_num=4)

     net = net_factory(net_type= NetName,
                       class_num=CLASS_NUM)
     net.load_state_dict(torch.load(model_path))
     mask=np.zeros((h_img,w_img))

     for r in range(rownum):
            for c in range(colnum):
                strXmin=float(c*colwidth*buchang)
                strYmin=float(r*rowheight*buchang)
                r1= r * rowheight*buchang
                r2= (r*buchang + 1) * rowheight
                if r2>h_img:
                    r2=h_img-1
                c1=c * colwidth*buchang
                c2=(c*buchang + 1) * colwidth
                if c2>w_img:
                    c2=w_img-1
                c2=(c*buchang + 1) * colwidth
                c1=int(c1)
                c2=int(c2)
                r1=int(r1)
                r2=int(r2)
                # print(c1,c2,r1,r2)
                if r1>=(h_img-1) or c1>=(w_img-1):
                    continue
                img=image_hb[r1:r2,c1:c2]
                img = BGR01(img)
                # img = rgb(img)

                maski=predict_img(net,img)

                if r1>0:
                    r1=int(r1+ rowheight/4)

                if c1>0:
                    c1=int(c1+ colwidth/4)
                
                if r1==0:
                    ri1=0
                else:
                    ri1= int(rowheight/4) 
                if c1==0:
                    ci1=0
                else:
                    ci1= int(colwidth/4)

                ri2= int(0.75*rowheight)
                ci2= int(0.75*colwidth)
                if c2 > (w_img ):
                    c2 = (w_img)
                if r2 > (h_img ):
                    r2 = (h_img)
                if c2<=(w_img- colwidth/4):
                    c2 = int(c2- colwidth/4) 
                else:
                    ci2=int(ci1+c2-c1) 
                if r2<=(h_img-rowheight/4):
                    r2 = int(r2- rowheight/4) 
                else:
                    ri2=int(ri1+r2-r1)        
                mask[r1:r2,c1:c2]=maski[ri1:ri2,ci1:ci2]
     result = tans_img(image, mask)
     return result
     # return mask

if __name__ == '__main__':

    model_path = 'D:/2022_yx/0606/0606_model/DeepLab_RGBD_best_model.pth'
    # Uncertainty_Aware_Mean_Teacher_3_labeled
    # Fully_Supervised_3_labeled
    images_path = 'D:/2022yanxing/20220919/22/dom'
    dsms_path= 'D:/2022yanxing/20220919/22/dsm'
    row=256
    col=256
    timeall = 0
    for idx, img_name in enumerate(sorted(os.listdir(images_path))):
        time0 = time.time()
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        img_path = os.path.join(images_path, img_name)
        dsm_path = os.path.join(dsms_path, img_name)
        print(img_path)

        # identify_img(filepath, weights_path, row, col, openMode)
        identify_datu(img_path,dsm_path,model_path)
        time1 = time.time()
        timed = time1 - time0
        timeall = timeall + timed
    print("time: ", timeall, " s")


