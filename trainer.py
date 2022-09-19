import torch
import SR_model
import utility
from utility import get_eye_patches
from decimal import Decimal
from tqdm import tqdm
import numpy as np

def angular(gaze, label):
    assert gaze.size == 3, "The size of gaze must be 3"
    assert label.size == 3, "The size of label must be 3"

    total = np.sum(gaze * label)
    return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def gazeto3d(gaze):
    assert gaze.size == 2, "The size of gaze must be 2"
    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt

class Trainer():
    def __init__(self, opt, loader, SR_model, Gaze_model, sr_loss, gaze_loss, ckp):
        self.opt = opt
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.SR_model = SR_model
        self.Gaze_model = Gaze_model
        self.sr_loss = sr_loss
        self.gaze_loss = gaze_loss
        self.optimizer = utility.make_optimizer(opt, self.SR_model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)

        self.gaze_optimizer = utility.make_gaze_optimizer(opt, self.Gaze_model)
        self.gaze_scheduler = utility.make_gaze_scheduler(opt, self.gaze_optimizer)

        self.error_last_1 = 1e8
        self.error_last = 1e8
        self.device = torch.device('cpu' if self.opt.cpu else 'cuda')


    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.sr_loss.start_log()
        self.gaze_loss.start_log()
        
        self.SR_model.train(False)
        self.Gaze_model.train()
        for name, param in self.SR_model.named_parameters():
            param.requires_grad = False
                
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, eye_coord, haedpose, gt_label, _) in enumerate(self.loader_train):
            timer_data.hold()
            timer_model.tic()
            accs = 0
            eval_psnr = 0
            self.optimizer.zero_grad()
            self.gaze_optimizer.zero_grad()

            # for param in self.SR_model.parameters():
            #     param.requires_grad=False

            lr, hr = lr.to(self.device), hr.to(self.device)
            SR_face = self.SR_model(lr)

            SR_face = utility.quantize(SR_face, self.opt.rgb_range)

            le, re = get_eye_patches(SR_face, eye_coord)
            data = {
                'face' : SR_face,
                'left' : le,
                'right' : re,
                'head_pose' : haedpose.to(self.device)
            }
            
            gazes = self.Gaze_model(data)
            srloss = self.sr_loss(SR_face, hr)

            gaze_loss = self.gaze_loss(gazes, gt_label.to(self.device))

            loss = srloss + gaze_loss

            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.gaze_optimizer.step()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            
            for k, gaze in enumerate(gazes):

                gaze = gaze.cpu().detach().numpy()
                gt = gt_label.cpu().numpy()[k]

                accs += angular(
                            gazeto3d(gaze),
                            gazeto3d(gt)
                        )
            eval_psnr += utility.calc_psnr(
                    SR_face, hr, self.opt.scale, self.opt.rgb_range,
                    benchmark=self.loader_test.dataset.benchmark
                )

                    
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t gaze_loss : {}\tBatch Ang_error{:.2f}\t{:.1f}+{:.1f}s SR_loss : {}\  batch PSNR{:.2f}'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.gaze_loss.display_loss(batch),
                    accs / self.opt.batch_size,
                    timer_model.release(),
                    timer_data.release(),
                    self.sr_loss.display_loss(batch),
                    eval_psnr
                    ))

            timer_data.tic()

        self.sr_loss.end_log(len(self.loader_train))
        self.gaze_loss.end_log(len(self.loader_train))
        self.error_last_1 = self.sr_loss.log[-1, -1]
        self.error_last = self.gaze_loss.log[-1, -1]
        self.step()

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.SR_model.eval()
        self.Gaze_model.eval()
        self.device = torch.device('cpu' if self.opt.cpu else 'cuda')
        timer_test = utility.timer()
        si = 0
        
        with torch.no_grad():
            accs = 0
            eval_psnr = 0
            eval_simm = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for batch, (lr, hr, eye_coord, haedpose, gt_label, filename) in enumerate(tqdm_test):
                filename = filename[0]
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                SR_face = self.SR_model(lr)
                SR_face = utility.quantize(SR_face, self.opt.rgb_range)
                
                le, re = get_eye_patches(SR_face, eye_coord)
                data = {
                    'face' : SR_face,
                    'left' : le,
                    'right' : re,
                    'head_pose' : haedpose.to(self.device)
                }
                gazes = self.Gaze_model(data)
                
                # save test results
                # if self.opt.save_results:
                
                eval_psnr += utility.calc_psnr(
                    SR_face, hr, self.opt.scale, self.opt.rgb_range,
                    benchmark=self.loader_test.dataset.benchmark
                )

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gt_label.cpu().numpy()[k]
                    angle = angular(gazeto3d(gaze),gazeto3d(gt))
                    # self.ckp.save_results_nopostfix(filename, SR_face, gaze)
                    accs += angle
                    
                    
                hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                sr_numpy = SR_face[0].cpu().numpy().transpose(1, 2, 0)
                simm = utility.SSIM(hr_numpy, sr_numpy)
                eval_simm += simm
            self.ckp.log[-1, si] = accs / len(self.loader_test)
            ang_error = accs / len(self.loader_test)
            best = self.ckp.log.min(0)
            self.ckp.write_log(
                '[{}]\tAngual error: {:.2f} (Best: {:.2f} @epoch {}), PSNR: {:.2f}, SSIM: {:.5f}'.format(
                    self.opt.data_test, 
                    ang_error,
                    best[0][si],
                    best[1][si] + 1,
                    eval_psnr / len(self.loader_test),
                    eval_simm / len(self.loader_test)
                )
            )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        self.gaze_scheduler.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs

