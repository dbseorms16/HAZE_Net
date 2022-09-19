import os
from data import srdata


class MPII(srdata.SRData):
    def __init__(self, args, name='MPII', train=True, benchmark=False):
        super(MPII, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(MPII, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'new_train')
        self.dir_lr = os.path.join(self.apath, 'LR')

        self.ext = ('', '.jpg')
