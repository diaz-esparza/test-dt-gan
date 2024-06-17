from lightning.pytorch.utilities.types import OptimizerLRScheduler

import lightning as L
import torch

from dataset import STAGE
import models

from typing import Any


class DefectModel(L.LightningModule):

    IN_CHANNELS:int=3
    FEATURE_ALLOC:int=64
    STYLE_DIMS:int=64
    LATENT_DIMS:int=16
    UPSAMPLING:str="bilinear"

    def __init__(
            self,
            num_domains:int=7,
            num_bg_classes:int=10,
            optimizer_fn:type[torch.optim.Optimizer]=torch.optim.Adam,
            lr:float=1e-4,
            l2:float=1e-6,
            )->None:
        super().__init__()
        self.__num_domains=num_domains
        self.__num_bg_classes=num_bg_classes
        self.optimizer_fn=optimizer_fn
        self.lr=lr
        self.l2=l2

        ## GENERATOR
        self.generator=models.Generator(
            in_channels=self.IN_CHANNELS,
            upsampling=self.UPSAMPLING,
            feature_alloc=self.FEATURE_ALLOC)

        ## MAPPING
        self.base_mapping=models.BaseMapping(
            latent_dims=self.LATENT_DIMS)
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
        self.disc_base=models.ImageEncoderLite(
            in_channels=self.IN_CHANNELS)
        ## BG AND DISC CLASSIFIERS
        self.disc_decoder,self.disc_classifier=models.StyleDecoder(
            num_domains=self.__num_domains,
            num_dims=1).expand()
        self.disc_bg_classifier=models.BGClassifier(
            num_classes=self.__num_bg_classes)
        ## FG CLASSIFIER
        self.disc_fg_preproc=models.DefectDecoder(
            num_domains=self.__num_domains,
            feature_alloc=self.FEATURE_ALLOC)
        self.disc_fg_classifier=models.FGClassifier(
            num_domains=self.__num_domains,
            feature_alloc=self.FEATURE_ALLOC)

    def shared_step(
            self,
            batch:tuple[
                torch.Tensor, ## Image
                torch.Tensor, ## Defect label
                torch.Tensor],## Background label...?
            stage:STAGE,
            batch_idx:int,
            )->dict[str,Any]:
        latent_code=torch.randn(self.LATENT_DIMS)
        latent_code/=latent_code.norm()


    def training_step(self,batch,batch_idx):
        return self.shared_step(batch,"train",batch_idx)
    def validation_step(self,batch,batch_idx):
        return self.shared_step(batch,"valid",batch_idx)
    def test_step(self,batch,batch_idx):
        return self.shared_step(batch,"test",batch_idx)

    def shared_epoch_end(self,stage:STAGE)->None:...

    def on_train_epoch_end(self):
        return self.shared_epoch_end("train")
    def on_validation_epoch_end(self):
        return self.shared_epoch_end("valid")
    def on_test_epoch_end(self):
        return self.shared_epoch_end("test")

    def configure_optimizers(self)->OptimizerLRScheduler:
        try:
            self.optimizer=self.optimizer_fn(
                self.parameters(),lr=self.lr,weight_decay=self.l2) ##type:ignore
        except TypeError as e:
            ## Check if it's a unexpected keyword argument error
            if ("'lr'" in str(e)) or ("'weight_decay'" in str(e)):
                raise TypeError("Optimizer does not support "
                                "typical parameters: {}".format(e))
            else: raise TypeError(e)

        return self.optimizer


