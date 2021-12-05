import numpy as np
import torch
import cv2
import math

def load_class_name(file):
    class_name = []
    with open(file,'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            class_name.append(line)
    return class_name

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
            # 该图片中的第j个被检查出的物体 box_
            box_ = box[j]
            x1 = int(box_[0] * width)
            y1 = int(box_[1] * height)
            x2 = int(box_[2] * width)
            y2 = int(box_[3] * height)

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

# people_boxes=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]
def get_people_stack(people_boxes,image,posenet_res):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]
    people_boxes_refine_for_all_image = []
    people_for_all_image = []
    for one_img_boxes in people_boxes:
        # 一张图片的所有人：one_img_boxes=[[x1,y1,x2,y2],[],...]
        one_img_boxes_refine =[]
        one_frame=[]
        for box in one_img_boxes:
            x1 = int(max(box[0]*width,0))
            y1 = int(max(box[1]*height,0))
            x2 = int(min(box[2]*width,width))
            y2 = int(min(box[3]*height,width))
            one_img_boxes_refine.append([max(0,box[0]),max(0,box[1]),min(1.0,box[2]),min(1.0,box[3])])
            if len(image.shape)==3:
                one_people = image[y1:y2+1,x1:x2+1]
                one_people = cv2.resize(one_people,(int(posenet_res[1]),int(posenet_res[0])))    # (h,w,3)
                # one_people = torch.from_numpy(one_people.transpose(2,0,1))
            one_frame.append(one_people)
        one_frame_array = np.stack(one_frame,axis=0)
        # people_for_all_image=[array(n_people,h,w,3) , ... ]
        people_for_all_image.append(one_frame_array)
        people_boxes_refine_for_all_image.append(one_img_boxes_refine)

    return people_for_all_image,people_boxes_refine_for_all_image
    

