# from networks.efficientunet import Effi_UNet
# from networks.enet import ENet
# from networks.pnet import PNet2D
#from networks.unet import UNet, UNet_DS, UNet_URPC
from networks.unet1 import UNet1, UNet_DS, UNet_URPC
from networks.unet import UNet
from networks.FCN8 import FCN8
from networks.deeplabv3_plus import DeepLab
from networks.deeplabv3_RGBD import DeepLab_RGBD
from networks.unet3 import  UNet_3Plus
from networks.UNet_2Plus import  UNet_2Plus

def net_factory(net_type="unet", in_chns=1, class_num=1):
    if net_type == "unet":
        net = UNet(n_channels=3, n_classes=class_num, bilinear=True).cuda()
    elif net_type == "FCN8":
        net = FCN8(num_classes=class_num).cuda()
    elif net_type == "DeepLab":
        net = DeepLab(num_classes=class_num).cuda()
    elif net_type == "DeepLab_RGBD":
        net = DeepLab_RGBD(num_classes=class_num).cuda()
    elif net_type == "unet2":
        net = UNet_2Plus(in_channels=in_chns,n_classes=class_num).cuda()
    elif net_type == "unet3":
        net = UNet_3Plus(in_channels=in_chns,n_classes=class_num).cuda()
    elif net_type == "unet1":
        net = UNet1(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "enet":
        net = ENet(in_channels=in_chns, num_classes=class_num).cuda()
    elif net_type == "unet_ds":
        net = UNet_DS(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "efficient_unet":
        net = Effi_UNet('efficientnet-b3', encoder_weights='imagenet',
                        in_channels=in_chns, classes=class_num).cuda()
    elif net_type == "pnet":
        net = PNet2D(in_chns, class_num, 64, [1, 2, 4, 8, 16]).cuda()
    else:
        net = None
    return net
