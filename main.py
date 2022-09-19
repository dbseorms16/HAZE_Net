import utility
import data
import model
import SR_model
import loss
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
from multiprocessing import Process, freeze_support

utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

def main():
    if checkpoint.ok:
        loader = data.Data(args)
        SR_m = SR_model.Model(args, checkpoint)
        sr_l = loss.SRLoss(args, checkpoint) if not args.test_only else None

        gaze_m = model.Model(args, checkpoint)
        ge_l = loss.Gaze_Loss(args, checkpoint) if not args.test_only else None
        
        t = Trainer(args, loader, SR_m, gaze_m, sr_l, ge_l, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
        checkpoint.done()

if __name__ == '__main__':
    freeze_support()
    main()