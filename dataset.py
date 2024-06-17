"""
Dataset architecture definition.

The average dataset architecture on my projects goes as follows:

    - Indexer: We offload __len__ and the ``(int) -> path``
    application in __getitem__ to this component.

    - Channel generator: This component manages the
    ``(path) -> data`` application. There can be more
    than one channel generator.

    - Regularizer: Mainly manages image (and possibly
    label) augmentations. If there was more than one
    channel generator, it also handles data concatenation.

    - Label generator: If necessary, manages
    ``(path) -> label`` application.

    - Timer: Self-explanatory, mainly for benchmarking
    and troubleshooting purposes.

If we want these components to be interchangeable, we
detach them from the Dataset class and place them as
__init__ args. 
"""

from abc import ABC, abstractmethod
from albumentations.pytorch import ToTensorV2
from collections.abc import Sequence
from os.path import join as jo
from PIL import Image
from numpy._typing import NDArray
from torch.utils.data import Dataset

import albumentations as A
import numpy as np
import numpy.typing as npt
import os
import os.path as osp
import random
import torch
import warnings

from typing import Any,Literal


ROOT=osp.dirname(__file__)
STORAGE=jo(ROOT,"storage")
SOURCE_DATA=jo(STORAGE,"sdi_dataset")
OUTPUTS=jo(STORAGE,"outputs","dataset_testing")
os.makedirs(OUTPUTS,exist_ok=True)
STAGE=Literal["train","valid","test"]

class Indexer(Sequence):
    @abstractmethod
    def __init__(self,root:str,stage:STAGE,n_normal:int=1500)->None:...

    @abstractmethod
    def __getitem__(self,idx:int)->tuple[str,int,int]:...


class ChannelGenerator(ABC):
    @abstractmethod
    def __call__(
        self,
        datapath:str,
        **kwargs:Any,
        )->np.ndarray:...

class Regularizer(ABC):
    LABEL_SUPPORT=False
    @abstractmethod
    def __call__(
            self,
            array:npt.NDArray[np.float32],
            )->torch.Tensor:...

class IndexerSDI(Indexer):
    r"""
    Sets a root directory and derives the item classes from
    the subdirectories.
    """
    FG_CLASSES={
        "normal":0,
        "scratches":1,
        "spots":2,
    }
    BG_CLASSES={
        "A":0,
        "B":1,
        "C":2,
    }

    ## For normal image exclusivity
    used_normal:list[str]=[]

    def __init__(self,root:str,stage:STAGE,n_normal:int=1500)->None:
        self.root=root

        self.items:dict[str,tuple[int,int]]={}
        self.list_items:list[str]=[]
        normal_items:list[tuple[str,int,int]]=[]

        for dir in os.scandir(self.root):
            if not dir.is_dir(): continue
            bg_class_raw,defect_raw=dir.name.split("_")
            bg_class=self.BG_CLASSES[bg_class_raw]
            defect=defect_raw=="nok"
            if defect:
                for d in os.scandir(jo(dir.path,stage.replace("valid","val"))):
                    fg_class=self.FG_CLASSES[d.name]
                    for i in os.scandir(d.path):
                        self.items[i.path]=(bg_class,fg_class)
                        self.list_items.append(i.path)
            else:
                for i in os.scandir(dir.path):
                    if i.path not in self.used_normal:
                        normal_items.append((i.path,bg_class,0))

        random.shuffle(normal_items)
        for i in normal_items[:n_normal]:
            self.items[i[0]]=i[1:]
            self.list_items.append(i[0])
            self.used_normal.append(i[0])

        if stage=="train": random.shuffle(self.list_items)

    def __len__(self)->int: return len(self.items)

    def __getitem__(self,idx:int)->tuple[str,int,int]:
        item=self.list_items[idx]
        return item,*self.items[item]


class ImageGenerator(ChannelGenerator):
    def __call__(
            self,
            datapath:str,
            )->np.ndarray:
        return np.array(Image.open(datapath))


class DummyRegularizer(Regularizer):
    def __init__(self)->None:
        self.transform=A.Compose([ToTensorV2()])

    def __call__(
            self,
            array:npt.NDArray[np.float32],
            )->tuple[torch.Tensor,int,int]:
        return self.transform(image=array)["image"]


class BasicRegularizer(Regularizer):
    def __init__(self)->None:
        self.transform=A.Compose([
            A.HorizontalFlip(),
            A.ColorJitter(),
            A.RandomBrightnessContrast(),
            ToTensorV2()])

    def __call__(
            self,
            array:npt.NDArray[np.float32],
            )->tuple[torch.Tensor,int,int]:
        return self.transform(image=array)["image"]


class DefectDataset(Dataset):
    def __init__(
            self,
            root:str,
            stage:STAGE,
            indexer:type[Indexer],
            generator:ChannelGenerator,
            regularizer:Regularizer,
            n_normal:int=1500,
            ) -> None:
        super().__init__()
        self.indexer=indexer(root,stage,n_normal)
        self.generator=generator
        self.regularizer=regularizer

    def __len__(self): return self.indexer.__len__()

    def __getitem__(self,idx:int)->tuple[torch.Tensor,int,int]:
        path,bg_class,fg_class=self.indexer[idx]
        item=self.generator(path)
        item_t=self.regularizer(item)
        return item_t,bg_class,fg_class


def __main():
    dataset=DefectDataset(
        SOURCE_DATA,
        "train",
        IndexerSDI,
        ImageGenerator(),
        BasicRegularizer())

    for f in os.scandir(OUTPUTS):
        if f.is_file(): os.remove(f.path)

    idxs=list(range(len(dataset)))
    random.shuffle(idxs)
    for i in idxs[:20]:
        img,bg_label,fg_label=dataset[i]
        Image.fromarray(img[0].numpy().__mul__(255).astype(np.uint8)).save(
            jo(OUTPUTS,f"{i}.{bg_label}{fg_label}.png"))

if __name__=="__main__": __main()

