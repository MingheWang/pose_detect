import sys
import os
import time
import math
import numpy as np

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


def sigmoid(x):
    return 1.0 / (np.exp(-x) + 1.)


def softmax(x):
    x = np.exp(x - np.expand_dims(np.max(x, axis=1), axis=1))
    x = x / np.expand_dims(x.sum(axis=1), axis=1)
    return x


def bbox_iou(box1, box2, x1y1x2y2=True):
    
    # print('iou box1:', box1)
    # print('iou box2:', box2)

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):         # boxes.shape=(num_i_j,4) , confs.shape=(num_i_j,)
    # print(boxes.shape) , shape=(num_i_j,4)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]       # 对该图片中的第i类所有置信度值由大到小排序 ，返回对应的下标排序

    keep = []
    while order.size > 0:
        idx_self = order[0]         # 当前框的下标
        idx_other = order[1:]       # 剩余框的下标shape=(num_n,)

        keep.append(idx_self)
        #———————————————————————————+++++++++++++++++++++++++++++++———————————————————————————————————
        # 计算当前框与剩余所有框的 IOU 值over
        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)
        # 计算当前框与剩余所有框的 IOU 值over
        #——————————————————————————++++++++++++++++++++++++++++++++——————————————————————————————————

        # np.where(condition)，输出满足条件(即非0)元素的坐标(等价于numpy.nonzero)，
        # 抑制与当前检测框iou大于num_thresh框
        inds = np.where(over <= nms_thresh)[0]          
        order = order[inds + 1]
    
    return np.array(keep)        # 返回留下的检测框下标



def plot_boxes_cv2(img, boxes, savename=None, class_names=None, color=None):
    # boxes = [ [[],[],...,[]] , [[],[],..,[]] ,....., [[],[],...,[]] ]，每个框表示为：[x1,y1,x2,y2，conf,conf,name_id]
    # 同一张图片中的所有检测框放在同一个列表中
    # len(bboxes_batch)等于图片数目，bboxes_batch是以图片为单位。
    import cv2
    img = np.copy(img)
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
        return int(r * 255)

    width = img.shape[1]
    height = img.shape[0]
    print('len of boxes:{}'.format(len(boxes)))
    for i in range(len(boxes)):
        # 第i张图片的所有实例的检测框 
        # box=[[],[],...,[]],每个框表示为一个实例的坐标：[x1,y1,x2,y2，conf,conf,name_id]
        box = boxes[i]   
        print(len(box))   # 在该图像中的所有被检测到的物体数量
        print(box)
        for j in range(len(box)):
            box_ = box[j]
            x1 = int(max(box_[0]*width,0))
            y1 = int(max(box_[1]*height,0))
            x2 = int(min(box_[2]*width,width))
            y2 = int(min(box_[3]*height,width))

            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box_) >= 7 and class_names:
                cls_conf = box_[5]
                cls_id = box_[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                # 将检测框的类别加在图像上
                img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, rgb, 1)
                img = cv2.putText(img,'{:.3f}'.format(box_[4]),(x2,y1),cv2.FONT_HERSHEY_SIMPLEX, 0.4, rgb, 1)
            # 将目标的检测绘制在图像上
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
    if savename:
        print("save plot results to %s" % savename)
        cv2.imwrite(savename, img)
    return img


def read_truths(lab_path):
    if not os.path.exists(lab_path):
        return np.array([])
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size / 5, 5)  # to avoid single truth problem
        return truths
    else:
        return np.array([])


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names



def post_processing(img, conf_thresh, nms_thresh, output):

    # anchors = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
    # num_anchors = 9
    # anchor_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # strides = [8, 16, 32]
    # anchor_step = len(anchors) // num_anchors
    """
        # output=[boxes, confs]
        # boxes: [batch, num1 + num2 + num3, 1, 4]
        # confs: [batch, num1 + num2 + num3, num_classes]
    """
    # box_array=[batch, num1 + num2 + num3, 1, 4]
    box_array = output[0]   
    # confs=[batch, num1 + num2 + num3, num_classes=80]
    confs = output[1]

    t1 = time.time()

    if type(box_array).__name__ != 'ndarray':
        box_array = box_array.cpu().detach().numpy()
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]

    # box_array=[batch, num1 + num2 + num3, 4]
    box_array = box_array[:, :, 0]
    # [batch, num1 + num2 + num3, num_classes] --> [batch, num1 + num2 + num3]
    max_conf = np.max(confs, axis=2)        # max_conf=[batch, num1 + num2 + num3]
    max_id = np.argmax(confs, axis=2)       # max_id=[batch, num1 + num2 + num3]

    t2 = time.time()

    bboxes_batch = []
    for i in range(box_array.shape[0]):    # box_array.shape = [batch, num1 + num2 + num3, 4]
        # 第i张图片
        """
        筛选检测框：
        1)先通过置信度阈值，对所有的检测框进行筛选
        2)再从满足置信值的目标框中选出第j类的所有目标框
        3)对该类的框进行nms非极大值抑制，得到该类满足条件的检测框
        """
        argwhere = max_conf[i] > conf_thresh        # argwhere=[num1 + num2 + num3,]
        # 筛选出该图片i中满足置信值的目标框
        l_box_array = box_array[i, argwhere, :]   # l_box_array.shape=(num_i,4)
        l_max_conf = max_conf[i, argwhere]   # shape=(num_i,)
        l_max_id = max_id[i, argwhere]       # shape=(num_i,)

        bboxes = []
        # nms for each class
        #----------------------------------------------------------------------------------------------------
        # person class
        #----------------------------------------------------------------------------------------------------
        # for j in range(num_classes):
        for j in range(num_classes):    # person class
            # 再从满足置信值的目标框中选出第j类的所有目标框cls_argwhere
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]                  # ll_box_array.shape=(num_i_j,4) , 第i张图的第j类的所有检测框
            ll_max_conf = l_max_conf[cls_argwhere]             # ll_max_conf.shape=(num_i_j,)
            ll_max_id = l_max_id[cls_argwhere]              # ll_max_id.shape=(num_i_j,)

            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh)     # 返回筛选出的该图像i的第j类检测框下标序列,shape=(num_i_j_k,)
            
            if (keep.size > 0):
                ll_box_array = ll_box_array[keep, :]        # ll_box_array.shape=(num_i_j_k,4)
                ll_max_conf = ll_max_conf[keep]             # ll_max_conf.shape=(num_i_j_k,)
                ll_max_id = ll_max_id[keep]                 # ll_max_id.shape=(num_i_j_k,)

                for k in range(ll_box_array.shape[0]):
                    # bboxes.append([x1,y1,x2,y2,conf,conf,name_j])
                    bboxes.append([ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3], ll_max_conf[k], ll_max_conf[k], ll_max_id[k]])
        # len(bboxes)最终等于该图像中的检测出的所有的类别的检测框总数目
        # 获得该图中该类所有满足的检测框列表[[x1,y1,x2,y2，conf,conf,name_j],[],..,[]]
        """
        同一张图片中的所有检测框放在同一个列表中
        len(bboxes_batch)等于图片总数，bboxes_batch是以图片为单位
        bboxes_batch = [ [[],[],...,[]] , [[],[],..,[]] ,....., [[],[],...,[]] ] ，每个框表示为：[x1,y1,x2,y2，conf,conf,name_id]
        """
        bboxes_batch.append(bboxes)          

    t3 = time.time()

    print('-----------------------------------')
    print('       max and argmax : %f' % (t2 - t1))
    print('                  nms : %f' % (t3 - t2))
    print('Post processing total : %f' % (t3 - t1))
    print('-----------------------------------')
    
    # bboxes_batch = [[[],[],...,[]],[[],[],..,[]],.....,[[],[],...,[]]]，每个框表示为：[x1,y1,x2,y2，conf,conf,name_id]
    # 同一张图片中的所有检测框放在同一个列表中
    # len(bboxes_batch)等于图片数目
    return bboxes_batch




