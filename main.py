import lightning as L
import torch

import models


class DefectModel(L.LightningModule):

    IN_CHANNELS:int=3
    FEATURE_ALLOC:int=64
    STYLE_DIMS:int=64
    UPSAMPLING:str="bilinear"

    def __init__(self,num_domains:int=7)->None:
        super().__init__()
        self.__num_domains=num_domains
        #torch.manual_seed(1234)

        ## GENERATOR
        self.generator=models.Generator(
            in_channels=self.IN_CHANNELS,
            upsampling=self.UPSAMPLING,
            feature_alloc=self.FEATURE_ALLOC)
        ## MAPPING
        self.base_mapping=models.BaseMapping()
        self.style_mapping=models.StyleMapping(
            num_domains=self.__num_domains,
            style_dims=self.STYLE_DIMS)
        self.defect_mapping=models.DefectMapping(
            num_domains=self.__num_domains,
            upsampling=self.UPSAMPLING,
            feature_alloc=self.FEATURE_ALLOC)
        ## STYLE_DEFECT ENCODER
        self.base_encoder=models.ImageEncoderLite(
            in_channels=self.IN_CHANNELS)
        self.style_decoder=models.StyleDecoder(
            num_domains=self.__num_domains,
            num_dims=self.STYLE_DIMS)
        self.feature_decoder=models.DefectDecoder(
            num_domains=self.__num_domains,
            feature_alloc=self.FEATURE_ALLOC)
        ## DISCRIMINATOR
        self.base_discriminator=models.ImageEncoderLite(
            in_channels=self.IN_CHANNELS)
        self.bg_discriminator=models.StyleDecoder(
            num_domains=self.__num_domains,
            num_dims=1)
        self.fg_discriminator=models.DefectDecoder(
            num_domains=self.__num_domains,
            feature_alloc=self.FEATURE_ALLOC)
        ## TODO: Add option to remove last StyleDecoder layer
        ## TODO: Add quick n' dirty linear nn for bg classifying
        ## TODO: Add 4 PreActResBlk into linear nn for fg classifying
