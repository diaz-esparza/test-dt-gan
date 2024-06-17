"""
Components for DT-GAN.
Reference:
    Ruyu Wang, Sabria Hoppe, Eduardo Monari, and Marco F. Huber.
    Defect Transfer GAN: Diverse Defect Synthesis for Data
    Augmentation. arXiv:2302.08366.
"""
from torch import nn

import os.path as osp
import torch

ROOT=osp.dirname(__file__)

__all__=[
    "ConvLayer",
    "AdaIN",
    "PreActResBlk",
    "Generator",
    "ImageEncoderLite",
    "StyleDecoder",
    "DefectDecoder",
    "BaseMapping",
    "StyleMapping",
    "DefectMapping",
]


class ConvLayer(nn.Module):
    """
    Template for layer-making
    TODO: Figure out noise placement
    """
    def __init__(
            self,
            convblock:nn.Module,
            resampler:nn.Module|None=None,
            norm:nn.Module|None=None,
            noise:bool=False,
            )->None:
        super().__init__()
        self.resblock=convblock
        self.resample=resampler
        self.norm=norm
        self.noise=noise

    def forward(
            self,
            x:torch.Tensor,
            style:torch.Tensor|None=None,
            )->torch.Tensor:
        for proc in [self.resblock,self.resample]:
            if proc is not None: x=proc(x)

        if self.norm is not None:
            if (style is not None) and (isinstance(self.norm,AdaIN)):
                x=self.norm(x,style)
            else: x=self.norm(x)

        if self.noise: x+=torch.randn_like(x)
        return x


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization

    Reference:
        Xun Huang and Serge Belongie, Arbitrary style transfer
        in real-time with adaptive instance normalization.
        In Proceedings of the IEEE International Conference on
        Computer Vision (ICCV), Oct 2017.
    """
    def __init__(self):
        super().__init__()
        self.IN=nn.LazyInstanceNorm2d()

    def forward(self,x:torch.Tensor,y:torch.Tensor)->torch.Tensor:
        std,mean=torch.std_mean(y)
        return std*self.IN(x)+mean




class PreActResBlk(nn.Module):
    """
    Pre-activation ResNet block

    Reference:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
        Identity mappings in deep residual networks.
        In Proceedings of the European Conference on Computer
        Vision (ECCV), pages 630-645. Springer, 2016.
    """
    def __init__(
            self,
            in_features:int,
            out_features:int|None=None,
            stride:int=1,
            )->None:
        super().__init__()
        if out_features is None: out_features=in_features

        self.bn1=nn.BatchNorm2d(in_features)
        self.conv1=nn.Conv2d(
            in_features,out_features,
            kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_features)
        self.conv2=nn.Conv2d(out_features,out_features,
            kernel_size=3,stride=stride,padding=1,bias=False)

        self.act=nn.ReLU()

    def forward(self,x:torch.Tensor)->torch.Tensor:
        identity=x
        x=self.bn1(x)
        x=self.act(x)
        x=self.conv1(x)
        x=self.bn2(x)
        x=self.act(x)
        x=self.conv2(x)

        x+=identity
        return x


class Generator(nn.Module):
    def __init__(self,in_channels:int=3,upsampling="bilinear",feature_alloc:int=64)->None:
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels,128,kernel_size=1),
            ConvLayer(PreActResBlk(128,256),nn.AvgPool2d(2),nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(256,512),nn.AvgPool2d(2),nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(512,512),nn.AvgPool2d(2),nn.LazyInstanceNorm2d()),

            ConvLayer(PreActResBlk(512,512),None,nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(512,512),None,nn.LazyInstanceNorm2d()),
        )
        self.__feature_alloc=feature_alloc
        bg_alloc=512-self.__feature_alloc
        self.bg_dec=nn.Sequential(
            ConvLayer(PreActResBlk(bg_alloc,bg_alloc),None,nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(bg_alloc,512),None,nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(512,512),None,nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(512,512),nn.Upsample(scale_factor=2,mode=upsampling),nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(512,256),nn.Upsample(scale_factor=2,mode=upsampling),nn.LazyInstanceNorm2d()),
            ConvLayer(PreActResBlk(256,448),nn.Upsample(scale_factor=2,mode=upsampling),nn.LazyInstanceNorm2d()),
        )
        ## We don't use sequential because of style transfer in  AdaIN
        self.fg_dec=nn.Sequential(
            ConvLayer(PreActResBlk(self.__feature_alloc,self.__feature_alloc),None,AdaIN()),
            ConvLayer(PreActResBlk(self.__feature_alloc,256),None,AdaIN()),
            ConvLayer(PreActResBlk(256,256),None,AdaIN()),
            ConvLayer(PreActResBlk(256,256),nn.Upsample(scale_factor=2,mode=upsampling),AdaIN()),
            ConvLayer(PreActResBlk(256,128),nn.Upsample(scale_factor=2,mode=upsampling),AdaIN()),
            ConvLayer(PreActResBlk(128,64),nn.Upsample(scale_factor=2,mode=upsampling),AdaIN()),
        )
        self.fuser=nn.Conv2d(512,3,kernel_size=1)

    def forward(
            self,
            x:torch.Tensor,
            style:torch.Tensor,
            defect:torch.Tensor|None=None,
            )->torch.Tensor:
        x=self.encoder(x)
        x,x_fg=torch.split(x,[512,512-self.__feature_alloc],dim=1)
        ## Defect replacement
        if defect is not None: x_fg=defect

        x=self.bg_dec(x)
        x_fg=self.fg_dec(x_fg,style)
        x=torch.cat((x,x_fg),dim=1)
        x=self.fuser(x)

        return x


class BaseMapping(nn.Module):
    def __init__(self,latent_dims:int=16)->None:
        super().__init__()
        self.model=nn.Sequential(*(
            [nn.Linear(latent_dims,512),nn.ReLU()]+
            [nn.Linear(512,512),nn.ReLU()]*3
        ))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.model(x)


class StyleMapping(nn.Module):
    ## KEPT AS PARAMETER CAUSE 
    def __init__(self,num_domains:int,style_dims:int=64)->None:
        super().__init__()
        self.model=nn.Sequential(*(
            [nn.Linear(512,512),nn.ReLU()]*3+
            [nn.Linear(512,style_dims*num_domains)]
        ))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.model(x)
        return x


class DefectMapping(nn.Module):
    def __init__(self,num_domains:int,upsampling="bilinear",feature_alloc:int=64)->None:
        super().__init__()
        self.model=nn.Sequential(
            ConvLayer(PreActResBlk(512,512),nn.Upsample(scale_factor=2,mode=upsampling),
                      nn.LazyInstanceNorm2d(),noise=True),
            ConvLayer(PreActResBlk(512,512),nn.Upsample(scale_factor=2,mode=upsampling),
                      nn.LazyInstanceNorm2d(),noise=True),
            ConvLayer(PreActResBlk(512,256),nn.Upsample(scale_factor=2,mode=upsampling),
                      nn.LazyInstanceNorm2d(),noise=True),
            ConvLayer(PreActResBlk(256,128),nn.Upsample(scale_factor=2,mode=upsampling),
                      nn.LazyInstanceNorm2d(),noise=True),
            ConvLayer(PreActResBlk(128,feature_alloc*num_domains),None,nn.LazyInstanceNorm2d(),noise=True),
        )

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.model(x)
        return x



class ImageEncoderLite(nn.Module):
    def __init__(self,in_channels:int=3)->None:
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=1),
            ConvLayer(PreActResBlk(64,256),nn.AvgPool2d(2)),
            ConvLayer(PreActResBlk(256,512),nn.AvgPool2d(2)),
            ConvLayer(PreActResBlk(512,512),nn.AvgPool2d(2)),
        )
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.encoder(x)


class StyleDecoder(nn.Module):
    """
    num_dims=style_dims=64 for style-defect decoder,
    num_dims=1 for discriminator (classificator).
    """
    def __init__(self,num_domains:int,num_dims:int)->None:
        super().__init__()
        ## We expect a 512x16x16 input
        self.decoder=nn.Sequential(
            ConvLayer(PreActResBlk(512,512),nn.AvgPool2d(2)),
            ConvLayer(PreActResBlk(512,512),nn.AvgPool2d(2)),
            nn.LeakyReLU(),
            nn.Conv2d(512,512,kernel_size=4),
            nn.LeakyReLU(),
        )
        self.classifier=nn.Linear(512,num_dims*num_domains)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.decoder(x)
        x=x.reshape((512,))
        x=self.classifier(x)
        return x


class DefectDecoder(nn.Module):
    def __init__(self,num_domains:int,feature_alloc:int=64)->None:
        super().__init__()
        self.decoder=nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(512,feature_alloc*num_domains,kernel_size=1),
        )
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.decoder(x)
        return x

