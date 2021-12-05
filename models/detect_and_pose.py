import torch
from torchvision.transforms import transforms
import numpy as np
import cv2
from detector_tools.torch_utils import do_detect
from models.darknet2pytorch import Darknet
from models.hrnet import HRNet
from detector_utils.utils import *
from pose_utils.utils import get_final_preds

class Detect_Pose:
    def __init__(self,args,cuda=False):
        self.args = args
        print(dir(self.args))
        self.cuda = cuda
        self.device = torch.device('cpu' if not cuda else 'cuda')
        self.detector = Darknet(args.yolov4_cfg)
        self.detector.load_weights(args.yolov4_weights)
        self.detector.to(self.device)
        self.detector.eval()
        self.posenet = HRNet(c=args.hrnet_c)
        self.posenet_res = args.hrnet_resolution
        self.posenet.load_state_dict(torch.load(args.hrnet_weights))
        self.posenet.to(self.device)
        self.posenet.eval()
        self.transforms_detect = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transforms_pose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.class_name = load_class_name(args.class_name_file)

    def predict(self,image):
        """
        detect object and human pose on a single image or a stack of n images.

        Args:
            image : (np.ndarray or torch.tensor)
        """
        self.orig_img = image
        # if len(image.shape)==3:
        #     image = cv2.resize(image,(self.detector.width,self.detector.height))
        #     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #     # image = np.transpose(image,(2,0,1))
        #     # assert image.shape[0]==3
        #     image = self.transforms_detect(image).unsqueeze(dim=0)
        # image = image.to(self.device)
        
        # boxes = [ [[],[],...,[]] , [[],[],...,[]] , .... , [[],[],...,[]] ] ，
        # 每个框表示为：[x1,y1,x2,y2，conf,conf,name_id],归一化坐标
        # 同一张图片中的所有检测框放在同一个列表中
        # len(boxes)等于图片数目，boxes是以图片为单位。
        boxes = do_detect(self.detector,image,self.args.confidence,self.args.nms_threshr,self.cuda)

        people_id = self.class_name.index('person')

        # people_boxes=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]
        # 人体检测框的归一化坐标,
        # people_keypoints_for_all_frame=[ np.ndarray(n_people,17,3) ,...],
        # 该坐标是节点在人体检测框上的相对坐标,0.0~1.0
        # people_boxes_refine_for_all_image=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]  , 0.0~1.0
        people_boxes,people_keypoints_for_all_frame,people_boxes_refine_for_all_image = self.get_people_keypoint(boxes,people_id)
        
        return boxes ,people_boxes, people_keypoints_for_all_frame,people_boxes_refine_for_all_image


    def get_people_keypoint(self,boxes,people_id):
        people_boxes = []
        for boxes_ in boxes:
            # 一张图片中的所有预测框: boxes_=[[x1,y1,x2,y2，conf,conf,name_id],[],...,[]]
            one_img_people_boxes = []
            for box in boxes_:
                # box = [x1,y1,x2,y2，conf,conf,name_id]
                if box[-1] == people_id:
                    one_img_people_boxes.append(box[:4])
            # people_boxes=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]
            # 人体检测框的归一化坐标
            people_boxes.append(one_img_people_boxes)
        # people_for_all_image=[array(n_people,h,w,3) , ... ]  , h=256或384 , w=192或288
        # people_boxes_refine_for_all_image=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]  , 0.0~1.0
        people_for_all_image,people_boxes_refine_for_all_image_0_1 = get_people_stack(people_boxes,self.orig_img,self.posenet_res)
        # people_keypoints_for_all_frame=[ np.ndarray(n_people,17,3),...] , 
        # 该坐标是节点在该人体检测框的中的位置(0.0~1.0) + conf
        people_keypoints_for_all_frame =  self.get_heatmaps_for_all_frame(people_for_all_image)

        return people_boxes,people_keypoints_for_all_frame,people_boxes_refine_for_all_image_0_1

    def get_heatmaps_for_all_frame(self,people_for_all_frame):
        # people_for_all_frame=[array(n_people,h,w,3) , ... ]  , h=256或384 , w=192或288
        people_keypoints_for_all_frame=[]
        for people_array_for_one_frame in people_for_all_frame:
            assert len(people_array_for_one_frame.shape)==4        # (n_people,h,w,3)
            #-------------------------------------------------------------------------------------
            # one frame
            #-------------------------------------------------------------------------------------
            one_frame_tensor = []
            for one_people_img_array in people_array_for_one_frame:
                # one_people_img_array=array(h,w,3)
                one_people_tensor = self.transforms_pose(one_people_img_array)          # one_people_tensor=tensor(3,h,w)
                one_frame_tensor.append(one_people_tensor)
            one_frame_stack = torch.stack(one_frame_tensor,dim=0)   # one_frame_stack=(n_people,3,h,w)

            ## outputs = tensor(n_people,17,64,48) , input.size=(n_people,3,256,192)
            ## outputs = tensor(n_people,17,96,72) , input.size=(n_people,3,384,288)
            output = self.posenet(one_frame_stack.to(self.device))
            out_put = output.detach().clone()
            out_put_cpu = out_put.cpu()
            out_numpy = out_put_cpu.numpy()
            """
            # 沿着根据热图上的值，某一方向微调整节点坐标
            # np.float64
            # keypoints_for_one_img=np.ndarry(n_people,17,2),(x,y)是热图尺度上的坐标
            # maxval=np.ndarray(n_people,17,1)
            """
            keypoints_for_one_img, maxvals = get_final_preds(out_numpy,adjust=False)

            keypoints = np.zeros((keypoints_for_one_img.shape[0],keypoints_for_one_img.shape[1],3))
            
            """
            # 将节点坐标归一化到 0.0~1.0,
            # coord_normalized,该坐标是节点在该人体检测框的中的位置
            """
            coord_normalized = keypoints_for_one_img/np.array([output.shape[-2],output.shape[-1]])
            keypoints[:,:,:2] = coord_normalized
            keypoints[:,:,2:] = maxvals
            people_keypoints_for_all_frame.append(keypoints)

        # people_keypoints_for_all_frame=[ np.ndarray(n_people,17,3),...]
        return people_keypoints_for_all_frame


        
