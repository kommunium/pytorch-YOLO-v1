import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import yoloDataset
from yoloLoss import yoloLoss

use_gpu = True
criterion = yoloLoss(7, 2, 5, 0.5, use_gpu)

file_root = 'JPEGImages/'
learning_rate = 0
num_epochs = 80
batch_size = 2

train_dataset = yoloDataset(root=file_root,
                            list_file='label_train.txt',
                            train=True,
                            transform=[transforms.ToTensor()])
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

test_dataset = yoloDataset(root=file_root,
                           list_file='label_test.txt',
                           train=False,
                           transform=[transforms.ToTensor()])
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=4,
                         pin_memory=True)
