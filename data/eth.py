import os
from data import srdata


class eth(srdata.SRData):
    def __init__(self, args, name='eth', train=True, benchmark=False):
        super(eth, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(eth, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'train')
        self.ext = ('', '.jpg')
