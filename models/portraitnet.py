import paddle
import paddle.nn as nn
from paddle.nn import layer
from .mobilenet_v2 import MobileNet

class backbone(MobileNet):
    def __init__(self, class_dim, scale, pretrained=None):
        super().__init__(class_dim=class_dim, scale=scale)
        self.channel_list=[16,24,32,96,320]
        if pretrained:
            self.set_state_dict(paddle.load(pretrained))
    
    def forward(self, inputs):
        feature_list=[]
        for i in range(1,9):
            inputs=getattr(self,'conv%d'%i)(inputs)
            if i in [2,3,4,6,8]:
                feature_list.append(inputs)
        return feature_list

def basic_conv(in_channels,out_channels,kernel_size,**kwargs):
    if 'act' in kwargs:
        act=kwargs.pop('act')
    else:
        act=nn.ReLU
    if 'norm' in kwargs:
        norm=kwargs.pop('norm')
    else:
        norm=nn.BatchNorm2D
    layerlist=[nn.Conv2D(in_channels,out_channels,kernel_size,**kwargs),norm(out_channels)]
    if act is not None:
        layerlist.append(act())
    return layerlist

class Dblock(nn.Layer):
    def __init__(self,in_channels,out_channels):
        super(Dblock,self).__init__()
        layerlist=\
            basic_conv(in_channels,in_channels,3,groups=in_channels,padding='same')+\
            basic_conv(in_channels,out_channels,1)+\
            basic_conv(out_channels,out_channels,3,groups=out_channels,padding='same')+\
            basic_conv(out_channels,out_channels,1,act=None)

        self.layers=nn.Sequential(*layerlist)
        self.shortcut=nn.Sequential(*basic_conv(in_channels,out_channels,1,act=None))

    def forward(self, inputs):
        return nn.functional.relu(self.layers(inputs)+self.shortcut(inputs))



class PortraitNet(nn.Layer):
    def __init__(self,num_classes=2,edge=True):
        super(PortraitNet,self).__init__()
        self.edge=edge
        self.backbone=backbone(1000,1.0)
        channel_list=self.backbone.channel_list
        channel_list.insert(0,8)
        for i in range(5):
            setattr(self,'upsample%d'%(i+1),
            nn.Sequential(
                        Dblock(channel_list[i+1],channel_list[i]),
                        nn.Conv2DTranspose(channel_list[i],channel_list[i],kernel_size=4,stride=2,padding=1)))
        self.mask_clas=nn.Conv2D(8,2,1)
        if self.edge:
            self.edge_clas=nn.Conv2D(8,2,1)
        
    def forward(self,inputs):
        feature_list=self.backbone(inputs)
        x=0
        for i in range(5):
            x=x+feature_list[4-i]
            x=getattr(self,'upsample%d'%(5-i))(x)
        if self.edge:
            return self.mask_clas(x),self.edge_clas(x)
        return self.mask_clas(x)


