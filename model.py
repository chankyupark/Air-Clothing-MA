''' model definition for clothing-ma
'''
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from roi_align.roi_align import RoIAlign


import cv2


def view_image(bboxes):
    ''' view image
    '''
    img = np.full((416, 416, 3), 100, dtype="uint8")
    for bbox in bboxes:
        c1 = tuple(bbox[0:2].int() * 16)
        c2 = tuple(bbox[2:4].int() * 16)
        cv2.rectangle(img, c1, c2, 128, 3)
    cv2.imshow("02", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Reshape(nn.Module):
    ''' Reshape
    '''

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        ''' forward operation
        '''
        return x.view(self.shape)


class EncoderClothing(nn.Module):
    ''' clothing-ma(multi-attributes) encoder
    '''

    def __init__(self, embed_size, device, pool_size, attribute_dim):
        """Load the pretrained yolo-v3 """
        super(EncoderClothing, self).__init__()
        self.device = device
        self.linear = nn.Linear(512 * pool_size * pool_size, embed_size)
        self.relu = nn.ReLU()
        # self.module_list = nn.ModuleList([nn.Linear(embed_size, att_size)
        # for att_size in attribute_dim])
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.5)
        self.pool_size = pool_size

        self.module_list = nn.ModuleList(
            [
                self.conv_bn(512, 256, 1, embed_size, att_size)
                for att_size in attribute_dim
            ]
        )

    def conv_bn(
        self,
        in_planes,
        out_planes,
        kernel_size,
        embed_size,
        att_size,
        stride=1,
        padding=0,
        bias=False,
    ):
        ''' each attributes encoding channel
        '''
        # "convolution with batchnorm, relu"
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, 1, stride=stride, padding=padding,
                bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.5),
            Reshape(-1, embed_size),
            nn.Linear(embed_size, att_size),
        )

    def forward(self, x):
        ''' forward operation
    '''
        outputs = {}

        for i in range(len(self.module_list)):
            output = self.module_list[i](x)
            outputs[i] = output

        return outputs



