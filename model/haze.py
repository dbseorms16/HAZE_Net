import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import save_image
from model import dct

def make_model(opt):
    print('Hazenet is maked')
    return Hazenet(opt)

class Hazenet(nn.Module):

    def __init__(self,opt):
        super(Hazenet, self).__init__()
        self.dct = dct.DCT_2D()
        self.idct = dct.IDCT_2D()

        _left_model = models.resnet18(pretrained=True)
        _right_model = models.resnet18(pretrained=True)
        _left_dct_model = models.resnet18(pretrained=True)
        _right_dct_model = models.resnet18(pretrained=True)
        _face_dct_model = models.resnet18(pretrained=True)
        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            _left_model.conv1,
            _left_model.bn1,
            _left_model.relu,
            _left_model.maxpool,
            _left_model.layer1,
            _left_model.layer2,
            _left_model.layer3,
            _left_model.layer4,
            _left_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_model.conv1,
            _right_model.bn1,
            _right_model.relu,
            _right_model.maxpool,
            _right_model.layer1,
            _right_model.layer2,
            _right_model.layer3,
            _right_model.layer4,
            _right_model.avgpool
        )
        self.left_dct_features = nn.Sequential(
            _left_dct_model.conv1,
            _left_dct_model.bn1,
            _left_dct_model.relu,
            _left_dct_model.maxpool,
            _left_dct_model.layer1,
            _left_dct_model.layer2,
            _left_dct_model.layer3,
            _left_dct_model.layer4,
            _left_dct_model.avgpool
        )
        self.right_dct_features = nn.Sequential(
            _right_dct_model.conv1,
            _right_dct_model.bn1,
            _right_dct_model.relu,
            _right_dct_model.maxpool,
            _right_dct_model.layer1,
            _right_dct_model.layer2,
            _right_dct_model.layer3,
            _right_dct_model.layer4,
            _right_dct_model.avgpool
        )
        self.face_dct_features = nn.Sequential(
            _face_dct_model.conv1,
            _face_dct_model.bn1,
            _face_dct_model.relu,
            _face_dct_model.maxpool,
            _face_dct_model.layer1,
            _face_dct_model.layer2,
            _face_dct_model.layer3,
            _face_dct_model.layer4,
            _face_dct_model.avgpool
        )

        self.totalFC1 = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True)
        )

        self.totalFC2 = nn.Sequential(
            nn.Linear(514, 256),
            nn.BatchNorm1d(256, momentum=0.99, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )
        self._init_weights()

    def forward(self, x_in):
        _,_, h, w = x_in['left'].size()
        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-(w//6)
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(x_in['left'].size())

        leftFeature = self.left_features(x_in['left'])
        leftFeature = leftFeature.view(leftFeature.size(0), -1)

        rightFeature = self.right_features(x_in['right'])
        rightFeature = rightFeature.view(rightFeature.size(0), -1)
        
        left_dct = self.dct(x_in['left'])
        left_dct = left_dct * hf_mask
        left_dct = self.idct(left_dct)
        leftdctfeature = self.left_dct_features(left_dct)
        leftdctfeature = leftdctfeature.view(leftdctfeature.size(0), -1)

        right_dct = self.dct(x_in['right'])
        right_dct = right_dct * hf_mask
        right_dct = self.idct(right_dct)
        rightdctfeature = self.right_dct_features(right_dct)
        rightdctfeature = rightdctfeature.view(rightdctfeature.size(0), -1)

        _,_, h, w = x_in['face'].size()
        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-(w//6)
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(x_in['face'].size())
        face_dct = self.dct(x_in['face'])
        face_dct = face_dct * hf_mask
        face_dct = self.idct(face_dct)
        facedctfeature = self.face_dct_features(face_dct)
        facedctfeature = facedctfeature.view(facedctfeature.size(0), -1)

        feature = torch.cat((leftFeature, rightFeature, leftdctfeature, rightdctfeature, facedctfeature), 1)

        feature = self.totalFC1(feature)
        feature = torch.cat((feature,  x_in['head_pose']), 1)
        gaze = self.totalFC2(feature)

        return gaze

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)