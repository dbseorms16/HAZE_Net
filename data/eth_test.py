
import os
from data import srdata


class eth_test(srdata.SRData):
    def __init__(self, args, name='eth_test', train=True, benchmark=False):
        super(eth_test, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(eth_test, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'val')
        self.ext = ('', '.jpg')
