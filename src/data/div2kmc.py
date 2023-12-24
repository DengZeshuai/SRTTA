import os
import glob
try:
    import imageio.v2 as imageio
except:
    import imageio
import torch
import numpy as np
from data import srdata, common

class DIV2KMC(srdata.SRData):
    def __init__(self, args, name='DIV2KMC', train=False, benchmark=False, corruption="GaussianBlur", lr_only=False):
        super(DIV2KMC, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
        self.corruption = corruption
        self.lr_only = lr_only
    
    def set_corruption(self, corruption="GaussianBlur"):
        self.corruption = corruption
        self._set_filesystem(self.args.dir_data, corruption)
        self.images_hr, self.images_lr = self._scan()
    
    def set_lr_only(self, lr_only=True):
        self.lr_only = lr_only
    
    def get_img_paths(self):
        return self.images_hr

    def _set_filesystem(self, dir_data, corruption='GaussianBlur'):
        if not hasattr(self, 'corruption'):
            self.corruption = corruption
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'gt')
        s = max(self.scale) if isinstance(self.scale, list) else self.scale
        self.dir_lr = os.path.join(self.apath, 'corruptions', self.corruption, 'X{}'.format(s))
        self.ext = ('.png', '.png')

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(self.dir_lr, filename + self.ext[1]))

        return names_hr, names_lr

    def __getitem__(self, idx):
        if self.lr_only:
            f_lr = self.images_lr[self.idx_scale][idx]
            filename = os.path.splitext(os.path.basename(f_lr))[0]
            lr = imageio.imread(f_lr)
            if lr.shape[2] > 3: lr = lr[:, :, :3]
            # convert torch to numpy
            lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
            lr = torch.from_numpy(lr).float()
            
            return lr, -1, filename
        else:
            lr, hr, filename = self._load_file(idx)
            pair = self.get_patch(lr, hr)
            pair = common.set_channel(*pair, n_channels=self.args.n_colors)
            pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)

        return pair_t[0], pair_t[1], filename