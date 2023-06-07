import numpy as np
import torch
#from medpy import metric
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,roc_curve,recall_score,auc,precision_recall_curve,average_precision_score,f1_score

from numpy import uint8
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        # print([intersection,union])
        IoU = intersection / (union+0.000001)  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def calculate_metric_percase(pred, gt,classes):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    #print(pred.shape)
    #x,y=pred.shape[0], pred.shape[1]
    #pred=pred.reshape(x*y,1)
    #gt=gt.reshape(x*y,1)
    metric = SegmentationMetric(classes)
    hist = metric.addBatch(pred, gt)
    pa = metric.pixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    pred = np.reshape(pred, newshape=(-1))
    gt = np.reshape(gt, newshape=(-1))
    # f1score=f1_score(gt, pred, average='macro')
    #print(pa,mIoU)
    if pred.sum() > 0:
        #dice=f1_score(gt, pred, average='macro')
        #precision=f1_score(gt, pred)
        #dice = 2*float(sum(uint8(pred[:] & gt[:]))) / float(sum(uint8(pred[:])) + sum(uint8(gt[:])));
        ##dice = metric.binary.dc(pred, gt)
        ##hd95 = metric.binary.hd95(pred, gt)
        #precision = float(sum(uint8(pred[:] & gt[:]))) / float(sum(uint8(pred[:])));
        return mIoU
    else:
        return 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    #image, label = image.squeeze(0).cpu().detach(
    #).numpy(), label.squeeze(0).cpu().detach().numpy()
    image, label = image.cpu().detach(
    ).numpy(), label.cpu().detach().numpy()
    prediction = np.zeros_like(label)
    #if len(label.shape)<len(image.shape) :
    #         label=torch.from_numpy(label).unsqueeze(0)
    #print(image.shape)
    #print(label.shape)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :,:]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y,1), order=0)
        
        #print(slice.shape)
        slice = np.transpose(slice, (2, 0, 1))
        input = torch.from_numpy(slice).unsqueeze(
            0).float().cuda()

        #print(label.shape)
        #print(input.shape)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            #print(out.shape)
            #print(np.max(out))
            #pred = out
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            #print(pred)
            prediction[ind] = pred
    metric_list = []
    #print(np.max(label))
    #print(np.max(prediction))
    #print(prediction.shape)
    for i in range(0, image.shape[0]):
        metric_list.append(calculate_metric_percase(
            prediction[i], label[i],classes))
    #for i in range(1, classes):
    #    metric_list.append(calculate_metric_percase(
    #        prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
