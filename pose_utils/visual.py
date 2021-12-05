import numpy as np
import cv2
from pose_utils.JOINTS import joints_dict


def add_joints(image, joints, color, dataset='coco'):
    h,w,_ = image.shape
    assert joints.shape[0] == len(joints_dict()[dataset]['keypoints'])
    # joints=ny.array(17,3)
    pair_orders = joints_dict()[dataset]["skeleton"]
    # add joints
    # joints=ny.array(17,3)
    for joint in joints:
        if joint[2] > 0:
            cv2.circle(image, (int(joint[0]), int(joint[1])), 1, color, 2)

    # add link
    for pair in pair_orders:
        if pair[0] < joints.shape[0] and pair[1] < joints.shape[0]:
            jointa = joints[pair[0]]
            jointb = joints[pair[1]]
            if jointa[2] > 0 and jointb[2] > 0:
                cv2.line(
                    image,
                    (int(jointa[0]*w), int(jointa[1]*h)), # (x,y)
                    (int(jointb[0]*w), int(jointb[1]*h)),   # (x,y)
                    color,
                    2
                )

    return image



# people_keypoints=np.array(n_people,17,3) , (x,y,conf) , 相对图片的归一化坐标
def draw_joint_and_skeleton(frame,people_keypoints,dataset='coco'):
    
    for person in people_keypoints:
        # person=np.array(17,3)
        color = np.random.randint(0,255,size=3)
        color = [int(i) for i in color]
        frame = add_joints(frame, person, color, dataset)

    return frame


