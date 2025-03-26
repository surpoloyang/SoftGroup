from .custom import CustomDataset
from glob import glob
import os.path as osp
import numpy as np
import torch

class PSDataset(CustomDataset):

    CLASSES = ('stem', 'leaf')

    def __init__(self, x4_split=False, **kwargs):
        super().__init__(**kwargs)
        self.x4_split = x4_split
            
    def get_filenames(self):
        if isinstance(self.prefix, str):
            self.prefix = [self.prefix]
        filenames_all = []
        for p in self.prefix:
            filenames = glob(osp.join(self.data_root, p + '*' + self.suffix))
            assert len(filenames) > 0, f'Empty {p}'
            filenames_all.extend(filenames)
        filenames_all = sorted(filenames_all * self.repeat)
        return filenames_all
    
    def load(self, filename):
        xyz, rgb, semantic_label, instance_label, _, _ = torch.load(filename)
        # subsample data
        if self.training and self.x4_split:
            N = xyz.shape[0]
            inds = np.random.choice(N, int(N * 0.25), replace=False)
            xyz = xyz[inds]
            rgb = rgb[inds]
            semantic_label = semantic_label[inds]
            instance_label = self.getCroppedInstLabel(instance_label, inds)
        return xyz, rgb, semantic_label, instance_label

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label
