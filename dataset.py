"""
txt image_name.jpg x y w h c x y w h c
"""

import os
import os.path
import random

import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data


def BGR2HSV(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)


def HSV2BGR(img):
    return cv.cvtColor(img, cv.COLOR_HSV2BGR)


def BGR2RGB(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def RandomBrightness(bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv.split(hsv)
        adjust = random.choice([0.5, 1.5])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def RandomSaturation(bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv.split(hsv)
        adjust = random.choice([0.5, 1.5])
        s *= adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def RandomHue(bgr):
    if random.random() < 0.5:
        hsv = BGR2HSV(bgr)
        h, s, v = cv.split(hsv)
        adjust = random.choice([0.5, 1.5])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv.merge((h, s, v))
        bgr = HSV2BGR(hsv)
    return bgr


def randomBlur(bgr):
    return cv.blur(bgr, (5,) * 2) if random.random() < 0.5 else bgr


def randomShift(bgr, boxes, labels):
    # 平移变换
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    if random.random() < 0.5:
        height, width, c = bgr.shape
        after_shift_image = np.zeros((height, width, c), dtype=bgr.dtype)
        after_shift_image[:, :, :] = (104, 117, 123)  # bgr
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)
        # print(bgr.shape,shift_x,shift_y)

        if shift_x >= 0 and shift_y >= 0:
            after_shift_image[int(shift_y):, int(shift_x):, :] = \
                bgr[:height - int(shift_y), :width - int(shift_x), :]
        elif shift_x >= 0 and shift_y < 0:
            after_shift_image[:height + int(shift_y), int(shift_x):, :] = \
                bgr[-int(shift_y):, :width - int(shift_x), :]
        elif shift_x < 0 and shift_y >= 0:
            after_shift_image[int(shift_y):, :width + int(shift_x), :] = \
                bgr[:height - int(shift_y), -int(shift_x):, :]
        elif shift_x < 0 and shift_y < 0:
            after_shift_image[:height + int(shift_y), :width + int(shift_x), :] = \
                bgr[-int(shift_y):, - int(shift_x):, :]

        shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
        center = center + shift_xy
        mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
        mask = (mask1 & mask2).view(-1, 1)
        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)

        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[int(shift_x),
                                        int(shift_y),
                                        int(shift_x),
                                        int(shift_y)]]).expand_as(boxes_in)
        boxes_in = boxes_in + box_shift
        labels_in = labels[mask.view(-1)]
        return after_shift_image, boxes_in, labels_in
    return bgr, boxes, labels


def randomScale(bgr, boxes):
    # fix height, scale width from 0.8 to 1.2
    if random.random() < 0.5:
        scale = random.uniform(0.8, 1.2)
        height, width, c = bgr.shape
        bgr = cv.resize(bgr, (int(width * scale), height))
        scale_tensor = torch.FloatTensor([[scale, 1] * 2]).expand_as(boxes)
        boxes *= scale_tensor
    return bgr, boxes


def randomCrop(bgr, boxes, labels):
    if random.random() < 0.5:
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        height, width, c = bgr.shape
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        x = random.uniform(0, width - w)
        y = random.uniform(0, height - h)
        x, y, h, w = int(x), int(y), int(h), int(w)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
        mask = (mask1 & mask2).view(-1, 1)

        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[x, y] * 2]).expand_as(boxes_in)

        boxes_in = boxes_in - box_shift
        boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
        boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
        boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
        boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

        labels_in = labels[mask.view(-1)]
        img_cropped = bgr[y:y + h, x:x + w, :]
        return img_cropped, boxes_in, labels_in
    return bgr, boxes, labels


def subMean(bgr, mean):
    mean = np.array(mean, dtype=np.float32)
    bgr = bgr - mean
    return bgr


def random_flip(im, boxes):
    if random.random() < 0.5:
        im_lr = np.fliplr(im).copy()
        h, w, _ = im.shape
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
        return im_lr, boxes
    return im, boxes


def random_bright(im, delta=16):
    alpha = random.random()
    if alpha > 0.3:
        im = im * alpha + random.randrange(-delta, delta)
        im = im.clip(min=0, max=255).astype(np.uint8)
    return im


def encoder(boxes, labels):
    """
    implement the encoder
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 14 x 14 x 30
    """

    grid_num = 14
    target = torch.zeros((grid_num,) * 2 + (30,))
    cell_size = 1. / grid_num
    wh = boxes[:, 2:] - boxes[:, :2]
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2

    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample / cell_size).ceil() - 1

        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1

        # relative up-left coordinates of matched cell
        xy = ij * cell_size
        delta_xy = (cxcy_sample - xy) / cell_size

        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
    return target


class yoloDataset(data.Dataset):
    image_size = 448

    def __init__(self, root, list_file, train, transform):
        cv.setNumThreads(0)
        cv.ocl.setUseOpenCL(False)

        print('data init')
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB

        # list_file = root + list_file
        print(list_file)

        with open(list_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            split = line.strip().split()
            self.fnames.append(split[0])
            num_boxes = (len(split) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x = float(split[1 + 5 * i])
                y = float(split[2 + 5 * i])
                x2 = float(split[3 + 5 * i])
                y2 = float(split[4 + 5 * i])
                c = split[5 + 5 * i]
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv.imread(os.path.join(self.root + fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        # if self.train:
        #     img, boxes = self.random_flip(img, boxes)
        #     img, boxes = self.randomScale(img, boxes)
        #     img = self.randomBlur(img)
        #     img = self.RandomBrightness(img)
        #     img = self.RandomHue(img)
        #     img = self.RandomSaturation(img)
        #     img, boxes, labels = self.randomShift(img, boxes, labels)
        #     img, boxes, labels = self.randomCrop(img, boxes, labels)

        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = subMean(img, self.mean)  # subtract the mean value
        img = cv.resize(img, (self.image_size,) * 2)
        target = encoder(boxes, labels)  # 7x7x30
        for t in self.transform:
            img = t(img)

        return img, target

    def __len__(self):
        return self.num_samples


def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = 'JPEGImages/'
    train_dataset = yoloDataset(root=file_root,
                                list_file='label_train.txt',
                                train=True,
                                transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    train_iter = iter(train_loader)
    for i in range(100):
        img, target = next(train_iter)
        print(img, target)


if __name__ == '__main__':
    main()
