import os
import glob
from data import common
import numpy as np
import imageio
import torch.utils.data as data
import pickle
import torch

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        
        self._set_filesystem(args.data_dir)
        self._get_imgs_path(args)
        self._set_dataset_length()

        with open(args.label, "rb") as f:
            label_dict = pickle.load(f)
      
        with open(args.eye_coord, "rb") as f:
            eye_coord = pickle.load(f)

        with open(args.face_coord, "rb") as f:
            face_coord = pickle.load(f)
        
        self.labels = label_dict
        self.eye_coord = eye_coord
        self.face_coord = face_coord
    
    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        
        lr, hr = common.set_channel(lr, hr, n_channels=self.args.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(
            lr, hr, rgb_range=self.args.rgb_range
        )

        label = self.labels[filename]
        eye_coord = self.eye_coord[filename]
        eye_coord_tensor = torch.from_numpy(np.array([int(eye_coord[0]),int(eye_coord[1]),int(eye_coord[2]),int(eye_coord[3])]))

        # ## mpii, eyediap
        headpose = torch.from_numpy(np.array([label[3], label[2]])).float()
        gaze_gt = torch.from_numpy(np.array([label[1], label[0]])).float()

        ## eth
        # haedpose = torch.from_numpy(np.array([label[1], label[0]])).float()
        # gaze_gt = torch.from_numpy(np.array([label[3], label[2]])).float()

        # f_coor = self.face_coord[filename]
        # f_coor = torch.tensor(f_coor).float()

        return lr_tensor, hr_tensor, eye_coord_tensor, headpose, gaze_gt, filename

    def __len__(self):
        return self.dataset_length

    def _get_imgs_path(self, args):
        list_hr, list_lr = self._scan()
        self.images_hr = list_hr
        self.images_lr = list_lr

    def _set_dataset_length(self):
        if self.train:
            self.dataset_length = self.args.test_every * self.args.batch_size
            repeat = self.dataset_length // len(self.images_hr)
            self.random_border = len(self.images_hr) * repeat
        else:
            self.dataset_length = len(self.images_hr)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = []
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            names_lr.append(os.path.join(
                self.dir_lr, 'x{}/{}{}'.format(
                    self.args.scale2, filename, self.ext[1]
                )
            ))

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.name)
        self.dir_hr = os.path.join(self.apath, 'val')
        self.ext = ('.jpg', '.jpg')

    def _get_index(self, idx):
        if self.train:
            if idx < self.random_border:
                return idx % len(self.images_hr)
            else:
                return np.random.randint(len(self.images_hr))
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        l_hr = self.images_lr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = imageio.imread(l_hr)

        return lr, hr, filename