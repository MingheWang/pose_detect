import cv2
import time
import argparse
from models.detect_and_pose import Detect_Pose
from detector_tools.utils import *
from detector_tools.torch_utils import *
from pose_utils.utils import adjust_keypoints
from pose_utils.visual import draw_joint_and_skeleton

def arg_parse():
    parser = argparse.ArgumentParser(description='yolov4 and posenet')
    parser.add_argument('--yolov4_cfg',default='detector_cfg/yolov4.cfg')
    parser.add_argument('--yolov4_weights',default='weights/detector/yolov4.weights')
    parser.add_argument('--class_name_file',default='data/coco.names')
    parser.add_argument('--conf_for_detector',dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh_for_detector", dest="nms_threshr", help="NMS Threshhold", default=0.4)
    parser.add_argument("--hrnet_resolution", help="Input resolution of the pose network.",default=(256,192), type=list)
    parser.add_argument('--hrnet_c',default=32)
    parser.add_argument('--hrnet_weights',default='weights/pose/pose_coco/pose_hrnet_w32_256x192.pth')
    return parser.parse_args()

if __name__=='__main__':
    args = arg_parse()
    cuda = torch.cuda.is_available()
    class_names = load_class_names(args.class_name_file)
    models = Detect_Pose(args,cuda)
    print(type(models))
    
    cap = cv2.VideoCapture('/home/wmh/pose_estimation/videos/test.mp4')
    while cap.isOpened():
        ret , frame = cap.read()
        if ret:
            start_time = time.time()
            """
            #------------------------------------------------------------------------
            boxes = [ [[],[],...,[]] , [[],[],...,[]] , .... , [[],[],...,[]] ] ，
            # 每个框表示为：[x1,y1,x2,y2，conf,conf,name_id],归一化坐标
            # 同一张图片中的所有检测框放在同一个列表中
            # len(boxes)等于图片数目，boxes是以图片为单位。
            # 此时，只有一张图片，所以len(boxes)=1
            #------------------------------------------------------------------------
            # people_boxes=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]
            # 人体检测框的归一化坐标,
            # len(people_boxes)等于图片数目
            # 此时，只有一张图片，所以len(people_boxes)=1
            #-------------------------------------------------------------------------
            # people_keypoints_for_all_frame=[ np.ndarray(n_people,17,3) ,...],
            # 该坐标是节点在人体检测框上的相对坐标,0.0~1.0
            # people_boxes_refine_for_all_image=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]  , 0.0~1.0
            #-------------------------------------------------------------------------
            """
            boxes,people_boxes,people_keypoints_for_all_frame,people_boxes_refine_for_all_image = models.predict(frame)
        
            # 调整节点坐标，将其由相对于检测框的调整到相对于图片
            # people_keypoints=np.array(n_people,17,3) , (x,y,conf) , 相对图片的归一化坐标
            people_keypoints = adjust_keypoints(people_boxes_refine_for_all_image,people_keypoints_for_all_frame)
            # 将目标的检测绘制在图像上 , 检测框的类别加在图像上
            orig_img = plot_boxes_cv2(frame, boxes, class_names=class_names)
            # 将人体节点绘制在图像上
            output_img = draw_joint_and_skeleton(orig_img,people_keypoints,'coco')
            fps = 1.0 / (time.time()-start_time)

            cv2.imshow('frame',output_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    
