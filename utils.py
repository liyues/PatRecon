import numpy as np
import os.path as osp

def save_record(output_path, epoch, train_loss, val_loss):
    filename = osp.join(output_path, 'loss_record.npz')
    np.savez(filename, epoch=epoch, train_loss=train_loss, val_loss=val_loss)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, lr_ratio, lr_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by lr_ratio every lr_decay epochs"""
    lr = lr * ((1.0 / lr_ratio) ** (epoch // lr_decay))

    # set learning rate for all parameters in the group
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
