import torch
import torch.nn as nn
from torchsummary import summary


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels) # 원 논문에서는 batchnorm 사용x
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.leakyrelu(out)

        return out
    

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20, in_channels=3):
        super(YOLOv1, self).__init__()
        self.in_channels = in_channels
        self.S, self.B, self.C = S, B, C
        self.architecture = [
            [1, (64, 7, 2, 3)], # [num_repeats, (out_channels, kernel_size, stride, padding)]
            'M',                # max pooling
            [1, (192, 3, 1, 1)],
            'M',
            [1, (128, 1, 1, 0)],
            [1, (256, 3, 1, 1)],
            [1, (256, 1, 1, 0)],
            [1, (512, 3, 1, 1)],
            'M',
            [4, (256, 1, 1, 0), (512, 3, 1, 1)],
            [1, (512, 1, 1, 0)],
            [1, (1024, 3, 1, 1)],
            'M',
            [2, (512, 1, 1, 0), (1024, 3, 1, 1)],
            [1, (1024, 3, 1, 1)],
            [1, (1024, 3, 2, 1)],
            [1, (1024, 3, 1, 1)],
            [1, (1024, 3, 1, 1)],
        ]
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(self.S, self.B, self.C)

        self._init_weights()

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == list:
                num_repeats = x[0]
                for _ in range(num_repeats):
                    for out_channels, kernel_size, stride, padding in x[1:]:
                        layers.append(CNNBlock(in_channels, out_channels, kernel_size, stride, padding))
                        in_channels = out_channels

            elif x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 1024), # 원 논문에서는 nn.Linear(1024 * S * S, 4096)
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, S * S * (5 * B + C)),
        )
    
    def _init_weights(self):
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

    def forward(self, x):
        out = self.darknet(x)
        out = self.fcs(torch.flatten(out, start_dim=1))
        out = out.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        return out
    

if __name__ == '__main__':
    # 모델 입출력 확인
    device = 'cuda'
    S, B, C = 7, 2, 20
    model = YOLOv1(S, B, C).to(device)
    x = torch.rand(2, 3, 448, 448).to(device)
    y = model(x)
    print(y.shape)

    # 모델 파라미터 확인
    summary(model, (3, 448, 448), batch_size=2)