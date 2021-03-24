import os

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from resnet_yolo import resnet50

# always index 0
CLASSES = ('part', 'center', 'side')

Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def decoder(pred):
    """
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    """

    grid_num = 14
    boxes = []
    cls_indices = []
    probs = []
    cell_size = 1. / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  # 7x7x30
    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    # mask1 = contain > 0.1  # 大于阈值
    mask1 = contain > 0.2  # 大于阈值
    mask2 = contain == contain.max()  # we always select the best contain_prob what ever it>0.9
    mask = (mask1 + mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    # print(i,j,b)
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell

                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]

                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    # print(float((contain_prob * max_prob)[0]))

                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_index = cls_index.resize_(1)

                        cls_indices.append(cls_index)
                        probs.append(contain_prob * max_prob)
                        # print(probs)
    print(len(boxes))
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indices = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)  # (n,4)
        probs = torch.cat(probs, 0)  # (n,)
        # print(probs)
        # print(cls_indexs)
        cls_indices = torch.cat(cls_indices, 0)  # (n,)
    keep = nms_(boxes, probs)
    print(len(boxes[keep]))
    return boxes[keep], cls_indices[keep], probs[keep]


def nms(bboxes, scores, threshold=0.5):
    """
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    """

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    order.resize_(1)
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def nms_(bboxes, scores, threshold=0.5):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)  # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)  # 降序排列

    keep = []
    while order.numel() > 0:  # torch.numel()返回张量元素个数
        if order.numel() == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])  # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)  # [N-1,]

        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx + 1]  # 修补索引之间的差值
    return torch.LongTensor(keep)  # Pytorch的索引值为LongTensor


#
# start predict one image
#
def predict_gpu(model, image_name, root_path=''):
    result = []
    image = cv.imread(root_path + image_name)
    h, w, _ = image.shape
    img = cv.resize(image, (448, 448))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor(), ])
    img = transform(img)
    # img = Variable(img[None, :, :, :], volatile=True)
    with torch.no_grad():
        img = Variable(img[None, :, :, :])
    img = img.cuda()

    pred = model(img)  # 1x7x7x30
    pred = pred.cpu()
    # print(pred)
    boxes, cls_indexs, probs = decoder(pred)
    print(cls_indexs)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), CLASSES[cls_index], image_name, prob])
    return result


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = resnet50()
    model.cuda()
    print('load model...')
    model.load_state_dict(torch.load('best.pth', map_location='cpu'))
    model.eval()
    # model.cpu()

    image_name = 'JPEGImages/0000190.jpg'
    image = cv.imread(image_name)
    print('predicting...')
    result = predict_gpu(model, image_name)
    print(result)

    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[CLASSES.index(class_name)]
        cv.rectangle(image, left_up, right_bottom, color, 2)

        label = class_name + str(round(prob, 2))
        text_size, baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])

        cv.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                     (p1[0] + text_size[0], p1[1] + text_size[1]),
                     color, -1)
        cv.putText(image, label, (p1[0], p1[1] + baseline),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

    cv.imwrite('result.jpg', image)
