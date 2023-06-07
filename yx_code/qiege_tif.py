import os
from PIL import Image
import cv2
from skimage import img_as_float, img_as_ubyte,exposure
import numpy as np
# from libtiff import TIFF
from scipy import misc
from osgeo import gdal


def splitimage(src, rowheight, colwidth, chongdie,dstpath):
    dataset = gdal.Open(src)
    img = dataset.ReadAsArray()#获取数据  
    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
                               
    #h, w = img.shape[1:3]
    #h, w = img.shape[0:2]
    num_bands = dataset.RasterCount
    h = dataset.RasterYSize
    w = dataset.RasterXSize
    buchang=1-chongdie
    rownum=int(h/(rowheight*buchang)) +1
    colnum=int(w/(colwidth*buchang)) +1

    if rownum <= h and colnum <= w:
        s = os.path.split(src)
        if dstpath == '':
            dstpath = s[0]
        fn = s[1].split('.')
        basename =  str(fn[0])[0:1]
        ext = str(fn[-1])
        num = 0        
        for r in range(rownum):
            for c in range(colnum):
                snum=   str(num)
                box = (c * colwidth*buchang, r * rowheight*buchang, (c*buchang + 1) * colwidth, (r*buchang + 1) * rowheight)
                r1= int(r * rowheight*buchang)
                r2= int((r*buchang + 1) * rowheight)
                c1=int(c * colwidth*buchang )
                c2=int((c*buchang + 1) * colwidth)
                #box=(0,0,100,100)
                rxmin=str(int(c*colwidth*buchang))
                rymin=str(int(r* rowheight*buchang))
                savepath= os.path.join(dstpath, basename + '_' + rxmin+ '_' + rymin+'.' + ext)
                if num_bands>1:
                    cropimage=img[:,r1:r2,c1:c2]
                else:
                    cropimage=img[r1:r2,c1:c2]
                driver = gdal.GetDriverByName("GTiff")
                dataset = driver.Create(savepath, rowheight, colwidth, num_bands, datatype)
                for i in range(num_bands):
                   if num_bands>1:
                        dataset.GetRasterBand(i+1).WriteArray(cropimage[i])
                   else:
                        dataset.GetRasterBand(i+1).WriteArray(cropimage)
                del dataset

                num = num + 1

        print('%s' % num)
    else:
        print('11')


def qiege(src,dstpath):
 if os.path.isfile(src):
     if (dstpath == '') or os.path.exists(dstpath):

         if rowheight > 0 and colwidth > 0:
             splitimage(src, rowheight, colwidth, 0.5,dstpath)
         else:
             print('22')
     else:
         print('%s' % dstpath)
 else:
     print('%s' % src)


if __name__ == "__main__":
    #inputImg="F:\\岩性\\3816\\3816mask.tif"
    #output="F:\\岩性\\3816\\3816mask_512"
    basename= '040'
    rowheight =256
    colwidth =256
    # inputImg = arcpy.GetParameterAsText(0)
    # output = arcpy.GetParameterAsText(1)
    # inputImg="H:\\yx2021\\data\\zhunnan3_Texture\\yangben\\img\\030mask.tif"
    # # basename = '040'
    # output="H:\\yx2021\\data\\zhunnan3_Texture\\yangben\\img\\train1024\\mask"
    # qiege(inputImg,output)


#####文件夹切割##########

    # images_path = 'H:/yx2021/data/zhunnan3_Texture/yangben/mask'
    # output = "H:\\yx2021\\data\\zhunnan3_Texture\\yangben\\train256\\mask"
    # for idx, img_name in enumerate(sorted(os.listdir(images_path))):
    #     if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    #         continue
    #     filepath = os.path.join(images_path, img_name)
    #     print(filepath)
    #     # identify_img(filepath, weights_path, row, col, openMode)
    #     qiege(filepath, output)

    images_path = 'F:/yx2021/sample/point1-sample/mask'
    output = "F:/yx2021/sample/202204/mask"
    for idx, img_name in enumerate(sorted(os.listdir(images_path))):
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            continue
        filepath = os.path.join(images_path, img_name)
        print(filepath)
        # identify_img(filepath, weights_path, row, col, openMode)
        qiege(filepath, output)
    # images_path = 'F:/yx2021/data/zhunnan3_Texture/yangben/mask'
    # output = "F:\\yx2021\\data\\zhunnan3_Texture\\yangben\\train256\\mask"
    # for idx, img_name in enumerate(sorted(os.listdir(images_path))):
    #     if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
    #         continue
    #     filepath = os.path.join(images_path, img_name)
    #     print(filepath)
    #     # identify_img(filepath, weights_path, row, col, openMode)
    #     qiege(filepath, output)


