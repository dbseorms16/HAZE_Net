import os
from data import srdata


class eyediap_test(srdata.SRData):
    def __init__(self, args, name='eyediap_test', train=True, benchmark=False):
        super(eyediap_test, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, data_dir):
        super(eyediap_test, self)._set_filesystem(data_dir)
        self.dir_hr = os.path.join(self.apath, 'val')
        self.dir_lr = os.path.join(self.apath, 'LR')

        self.ext = ('', '.jpg')
