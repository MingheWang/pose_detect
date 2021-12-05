import math
import numpy as np

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    # 将每类节点热图由二维展开为一维heatmaps_reshaped.shape=(b,17,h*w)
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))            
    # 17类节点中，对应的热图共17层，因为每一层的热图最多只有一个几点，所以只选择值最大的点
    idx = np.argmax(heatmaps_reshaped, 2)       # 每一层热图的最大值点在h*w中的索引 , idx.shape=(b,17)
    maxvals = np.amax(heatmaps_reshaped, 2)     # 每一层热图的最大值 , maxvals.shape=(b,17)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))      # maxvals.shape=(b,17,1)
    idx = idx.reshape((batch_size, num_joints, 1))         # idx.shape=(b,17,1)

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)     # idx延着第3维重复2次 , preds.shape=(b,17,2)
    # 将一维索引再恢复为二维坐标(x,y)
    preds[:, :, 0] = (preds[:, :, 0]) % width    # x
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)     # y

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))        # mask屏蔽掉置信度小于0的节点
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask      # mask屏蔽掉置信度小于0的节点
    # numpy
    # preds.shape=(b,17,(x,y))
    # maxvals.shape=(b,17,1)
    return preds, maxvals


def get_final_preds( batch_heatmaps,adjust=False ):
    coords, maxvals = get_max_preds(batch_heatmaps)    # coords=np.array(b,17,(x,y))   # maxvals=np.array(b,17,1)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    """
    沿着根据热图上的值，某一方向微调整节点坐标
    """
    if adjust:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]       # shape=(h,w)
                px = int(math.floor(coords[n][p][0] + 0.5))     # x+0.5
                py = int(math.floor(coords[n][p][1] + 0.5))     # y+0.5
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    # 沿某一方向微调节点坐标
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # preds.shape=(b,17,[x,y])
    # maxvals.shape=(b,17,1)
    return preds, maxvals


def adjust_keypoints(people_boxes,people_keypoints_for_all_frame):
    """
    #------------------------------------------------------------------------
    # people_boxes=[ [[x1,y1,x2,y2],[],...] , [[x1,y1,x2,y2],[],...] , ... ]
    # 人体检测框的归一化坐标,
    # len(people_boxes)等于图片数目
    # 此时，只有一张图片，所以len(people_boxes)=1
    #-------------------------------------------------------------------------
    # people_keypoints_for_all_frame=[ np.ndarray(n_people,17,3) ,...],(x,y,conf)
    # 该坐标是节点在人体检测框上的相对坐标,0.0~1.0
    #-------------------------------------------------------------------------
    """
    assert len(people_boxes)==1
    assert len(people_keypoints_for_all_frame)==1

    boxes = np.array(people_boxes[0])           # boxes.shape=(n_people,4)
    people_keypoints = people_keypoints_for_all_frame[0]        # people_keypoints.shape=(n_people,17,3) , (x,y,conf)

    assert boxes.shape[0]==people_keypoints.shape[0]

    boxes_x1 = boxes[:,0]       # boxes_x1.shape=(n_people,)
    boxes_x1 = np.tile(boxes_x1,(17,1)).T       # boxes_x1.shape=(n_people,17)
    boxes_y1 = boxes[:,1]
    boxes_y1 = np.tile(boxes_y1,(17,1)).T
    boxes_x2 = boxes[:,2]
    boxes_x2 = np.tile(boxes_x2,(17,1)).T
    boxes_y2 = boxes[:,3]
    boxes_y2 = np.tile(boxes_y2,(17,1)).T

    kpt_x = people_keypoints[:,:,0]         # kpt_x.shape=(n_people,17)
    kpt_y = people_keypoints[:,:,1]         

    keypoints = np.zeros(people_keypoints.shape)        # shape=(n_people,17,3) , (x,y,conf)

    keypoints[:,:,0] = boxes_x1+(boxes_x2-boxes_x1)*kpt_x        # x
    keypoints[:,:,1] = boxes_y1+(boxes_y2-boxes_y1)*kpt_y        # y
    keypoints[:,:,2] = people_keypoints[:,:,2]

    return keypoints


