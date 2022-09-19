import math
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import cv2
from skimage.metrics import structural_similarity as ssim 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    
def SSIM(original, compressed): 
    # Convert the images to grayscale
    grayA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(compressed, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8") 
    # 6. You can print only the score if you want
    # print(f"SSIM value is {score}")
    return score


def init_model(args):
    # Set the templates here
    if args.scale == 4:
        args.n_blocks = 40
        args.n_feats = 20
    elif args.scale == 8:
        args.n_blocks = 36
        args.n_feats = 10
    else:
        print('Use defaults n_blocks and n_feats.')
    args.dual = True


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    if sr.size(-2) > hr.size(-2) or sr.size(-1) > hr.size(-1):
        print("the dimention of sr image is not equal to hr's! ")
        sr = sr[:,:,:hr.size(-2),:hr.size(-1)]
    diff = (sr - hr).data.div(rgb_range)

    # if benchmark:
    #     shave = scale
    #     if diff.size(1) > 1:
    #         convert = diff.new(1, 3, 1, 1)
    #         convert[0, 0, 0, 0] = 65.738
    #         convert[0, 1, 0, 0] = 129.057
    #         convert[0, 2, 0, 0] = 25.064
    #         diff.mul_(convert).div_(256)
    #         diff = diff.sum(dim=1, keepdim=True)
    # else:
    shave = scale[-1] + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def make_optimizer(opt, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {
        'betas': (opt.beta1, opt.beta2),
        'eps': opt.epsilon
    }
    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_gaze_optimizer(opt, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {
        'betas': (opt.beta1, opt.beta2),
        'eps': opt.epsilon
    }
    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    
    return optimizer_function(trainable, **kwargs)

def make_scheduler(opt, my_optimizer):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        float(opt.epochs),
        eta_min=opt.eta_min
    )

    return scheduler

def make_gaze_scheduler(opt, my_optimizer):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        float(opt.epochs),
        eta_min=opt.eta_min
    )

    return scheduler

def get_eye_patches(SR, eye_coords):
    b = SR.size(0)
    le = torch.zeros_like(torch.empty(b,3,36,60)).to('cuda:0')
    re = torch.zeros_like(torch.empty(b,3,36,60)).to('cuda:0')

    for i in range(b):
        face = SR[i]
        eye_coord = eye_coords[i] 

        left_x = eye_coord[0]
        left_y = eye_coord[1]
        right_x =eye_coord[2]
        right_y =eye_coord[3]

        left_eye = face[:, abs(left_y-18):left_y+18, abs(left_x-30):left_x + 30]
        right_eye = face[:, abs(right_y-18):right_y+18, abs(right_x-30):right_x + 30]

        if left_eye.size() != (3, 36, 60):
            shift = left_x-30
            yshift = 36 - left_eye.size(1)
            if abs(left_y-yshift)-18 < 0 :
                left_y = 18
                yshift = 0 

            left_eye = face[:, (abs(left_y-yshift))-18:(abs(left_y-yshift))+18, (abs(left_x-shift))-30:(abs(left_x-shift)) + 30]

        if right_eye.size() != (3, 36, 60):
            shift = right_x-30
            yshift = 36 - right_eye.size(1)

            if abs(right_y-yshift)-18 < 0 :
                right_y = 18
                yshift = 0 

            right_eye = face[:, (abs(right_y-yshift))-18:(abs(right_y-yshift))+18, (abs(right_x-shift))-30:(abs(right_x-shift)) + 30]
        
        le[i] = left_eye
        re[i] = right_eye

    return le, re

