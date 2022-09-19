import os
from data import srdata


class eyediap(srdata.SRData):
    def __init__(self, args, name='eyediap', train=True, benchmark=False):
        super(eyediap, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(eyediap, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'train')
        self.dir_lr = os.path.join(self.apath, 'LR')

        self.ext = ('', '.jpg')
