import os
import torch
import datetime
import numpy as np
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if opt.save == '.': opt.save = '../experiment/EXP/' + now
        self.dir = opt.save

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.dir)
        _make_dir(self.dir + '/model')
        _make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(opt):
                f.write('{}: {}\n'.format(arg, getattr(opt, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        # trainer.model.save(self.dir, is_best=is_best)
        trainer.SR_model.save(self.dir, is_best=is_best)
        trainer.Gaze_model.save(self.dir, is_best=is_best)
        trainer.sr_loss.save(self.dir)
        trainer.gaze_loss.save(self.dir)
        trainer.sr_loss.plot_loss(self.dir, epoch)
        trainer.gaze_loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(
            trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'optimizer.pt')
        )
        torch.save(
            trainer.gaze_optimizer.state_dict(),
            # trainer.optimizer.state_dict(),
            os.path.join(self.dir, 'gaze_optimizer.pt')
        )
    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        label = 'Angualr error on {}'.format(self.opt.data_test)
        fig = plt.figure()
        plt.title(label)
        plt.plot(
            axis,
            self.log[:, 0].numpy(),
            label='Angualr error'
        )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Angualr error')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.opt.data_test))
        plt.close(fig)
    
    def save_results_nopostfix(self, filename, sr, gaze, face_center):
        apath = '{}/results/{}'.format(self.dir, self.opt.data_test)
        if not os.path.exists(apath):
            os.makedirs(apath)
        filename = os.path.join(apath, filename)

        normalized = sr[0].data.mul(255 / self.opt.rgb_range)
        ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy().copy()
        ndarr = self.draw_gaze_eye(ndarr,  gaze, [face_center[0], face_center[1]], thickness=3, color=(255, 0, 0) )  # draw gaze direction on the face patch image
        imageio.imwrite('{}.jpg'.format(filename), ndarr)


    
    def pitchyaw_to_vector(self, pitchyaws):
        r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

        Args:
            pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

        Returns:
            :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
        """
        n = pitchyaws.shape[0]
        sin = np.sin(pitchyaws)
        cos = np.cos(pitchyaws)
        out = np.empty((n, 3))
        out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
        out[:, 1] = sin[:, 0]
        out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
        return out

    def draw_gaze(self, image_in, length , pitchyaw, thickness=2, color=(0, 0, 255)):
        """Draw gaze angle on given image with a given eye positions."""
        pitchyaw = pitchyaw[0].cpu()
        image_out = image_in
        (h, w) = image_in.shape[:2]
        pos = (int(h / 2.0), int(w / 2.0))
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[1])
        cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)
        return image_out
    
    def draw_gaze_eye(self, image_in, length , pitchyaw, eye_coord, color, thickness=2):
        """Draw gaze angle on given image with a given eye positions."""
        pitchyaw = pitchyaw[0].cpu()
        image_out = image_in
        (h, w) = image_in.shape[:2]
        pos = (int(eye_coord[0]), int(eye_coord[1]))
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[1])
        cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                    thickness, cv2.LINE_AA, tipLength=0.2)
        return image_out