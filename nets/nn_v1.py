import torch
import torch.nn as nn
from utils.utils_v1 import GlobalAvgPool2d, Conv, Flatten


# Used to image classification backbone
class BACKBONE(nn.Module):
    def __init__(self, num_classes=1000, init_weight=True):
        super(BACKBONE, self).__init__()

        # input: 3 x 448 x 448
        self.extractor = nn.Sequential(
            Conv(3, 64, 7, 2),
            nn.MaxPool2d(2, 2),
            # 64 x 112 x 112

            Conv(64, 192, 3),
            nn.MaxPool2d(2, 2),
            # 192 x 56 x 56

            Conv(192, 128, 1),
            Conv(128, 256, 3),
            Conv(256, 256, 1),
            Conv(256, 512, 3),
            nn.MaxPool2d(2, 2),
            # 512 x 28 x 28

            Conv(512, 256, 1),
            Conv(256, 512, 3),
            # 512 x 28 x 28

            Conv(512, 256, 1),
            Conv(256, 512, 3),
            # 512 x 28 x 28

            Conv(512, 256, 1),
            Conv(256, 512, 3),
            # 512 x 28 x 28

            Conv(512, 256, 1),
            Conv(256, 512, 3),
            # 512 x 28 x 28

            Conv(512, 512, 1),
            Conv(512, 1024, 3),
            nn.MaxPool2d(2, 2),
            # 1024 x 14 x 14

            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
            Conv(1024, 512, 1),
            Conv(512, 1024, 3),
            # 1024 x 14 x 14
        )
        # output: 1024 x 14 x 14

        # Classifier to predict the class
        # pack the feature extractor and the classifier
        self.classifier = nn.Sequential(
            *self.extractor,
            # 1024 x 14 x 14
            GlobalAvgPool2d(),
            # 1024 x 1 x 1
            nn.Linear(1024, num_classes)
            # num_classes
        )

        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class HEAD(nn.Module):
    def __init__(self, fs, nb, nc):
        super(HEAD, self).__init__()

        # input: 1024 x 14 x 14
        self.conv = nn.Sequential(
            Conv(1024, 1024, 3),
            Conv(1024, 1024, 3, 2),
            Conv(1024, 1024, 3),
            Conv(1024, 1024, 3)
        )
        # output: 1024 x 7 x 7

        # input: 1024 x 7 x 7
        self.detect = nn.Sequential(
            Flatten(), # -> [1024 * 7 * 7,]
            nn.Linear(1024 * 7 * 7, 4096),  # [1024 * 7 * 7,] -> [4096,]
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, fs * fs * (5 * nb + nc)),  # 4096 -> S x S x (5 * B + C)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.detect(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self, fs=7, nb=2, nc=20, pretrained_backbone=False):
        super(YOLOv1, self).__init__()

        self.FS = fs
        self.NB = nb
        self.NC = nc
        
        if pretrained_backbone:
            self.features = BACKBONE().extractor
            darknet = BACKBONE()
            darknet = nn.DataParallel(darknet)
            src_state_dict = torch.load('model_best.pth.tar')['state_dict']
            dst_state_dict = darknet.state_dict()

            for k in dst_state_dict.keys():
                print('Loading weight of', k)
                dst_state_dict[k] = src_state_dict[k]
                
            darknet.load_state_dict(dst_state_dict)
            self.features = darknet.module.extractor
        else:
            self.features = BACKBONE().extractor

        self.head = HEAD(self.FS, self.NB, self.NC)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)

        x = x.view(-1, self.FS, self.FS, 5 * self.NB + self.NC)
        return x # [batch_size, S, S, 5 * B + C]


if __name__ == '__main__':
    yolo = YOLOv1()

    # Test image
    # batch size * channel * height * width = [2, 3, 448, 448]
    image = torch.randn(2, 3, 448, 448)
    output = yolo(image)
    print(output.size())  # torch.Size([2, 7, 7, 30])