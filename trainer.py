import shutil
import os.path as osp
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import ReconNet
from utils import AverageMeter

class Trainer_ReconNet(nn.Module):
    def __init__(self, args):
        super(Trainer_ReconNet, self).__init__()

        self.exp_name = args.exp
        self.arch = args.arch
        self.print_freq = args.print_freq
        self.output_path = args.output_path
        self.resume = args.resume
        self.best_loss = 1e5

        # create model
        print("=> Creating model...")
        if self.arch == 'ReconNet':
            self.model = ReconNet(in_channels=args.num_views, out_channels=args.output_channel, gain=args.init_gain, init_type=args.init_type)
            self.model = nn.DataParallel(self.model).cuda()
        else:
            assert False, print('Not implemented model: {}'.format(self.arch))

        # define loss function
        if args.loss == 'l1':
            # L1 loss
            self.criterion = nn.L1Loss(size_average=True, reduce=True).cuda() 
        elif args.loss == 'l2':
            # L2 loss (mean-square-error)
            self.criterion = nn.MSELoss(size_average=True, reduce=True).cuda()
        else:
            assert False, print('Not implemented loss: {}'.format(args.loss))

        # define optimizer
        if args.optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                            lr=args.lr,
                                            betas=(0.5, 0.999),
                                            weight_decay=args.weight_decay,  
                                            )
        else:
            assert False, print('Not implemented optimizer: {}'.format(args.optim))



    def train_epoch(self, train_loader, epoch):

        train_loss = AverageMeter()

        # train mode
        self.model.train()

        for i, (input, target) in enumerate(train_loader):

            input_var, target_var = Variable(input), Variable(target)
            input_var = input_var.cuda()
            target_var = target_var.cuda()

            # compute output
            output = self.model(input_var)

            # compute loss
            loss = self.criterion(output, target_var)
            train_loss.update(loss.data.item(), input.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # display info
            if i % self.print_freq == 0:
                print('Epoch: [{0}] \t'
                      'Iter: [{1}/{2}]\t'
                      'Train Loss: {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                       epoch, i, len(train_loader), 
                       loss=train_loss))

        # finish current epoch
        print('Finish Epoch: [{0}]\t'
              'Average Train Loss: {loss.avg:.5f}\t'.format(
               epoch, loss=train_loss))

        return train_loss.avg


    def validate(self, val_loader):

        val_loss = AverageMeter()
        batch_time = AverageMeter()

        # evaluation mode
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            input_var, target_var = Variable(input), Variable(target)
            input_var = input_var.cuda()
            target_var = target_var.cuda()

            # compute output
            output = self.model(input_var)

            # compute loss
            loss = self.criterion(output, target_var)
            val_loss.update(loss.data.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time()-end)
            end = time.time()

            # if i % args.print_freq == 0:
            print('Val: [{0}/{1}]\t'
                  'Time {batch_time.val: .3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                   i, len(val_loader), 
                   batch_time=batch_time, 
                   loss=val_loss))

        return val_loss.avg


    def save(self, curr_val_loss, epoch):
        # update best loss and save checkpoint
        is_best = curr_val_loss < self.best_loss
        self.best_loss = min(curr_val_loss, self.best_loss)

        state = {'epoch': epoch + 1,
                'arch': self.arch,
                'state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'optimizer': self.optimizer.state_dict(),
                }

        filename = osp.join(self.output_path, 'curr_model.pth.tar')
        best_filename = osp.join(self.output_path, 'best_model.pth.tar')

        print('! Saving checkpoint: {}'.format(filename))
        torch.save(state, filename)

        if is_best:
            print('!! Saving best checkpoint: {}'.format(best_filename))
            shutil.copyfile(filename, best_filename)


    def load(self):

        if self.resume == 'best':
            ckpt_file = osp.join(self.output_path, 'best_model.pth.tar')
        elif self.resume == 'final':
            ckpt_file = osp.join(self.output_path, 'curr_model.pth.tar')
        else:
            assert False, print("=> no available checkpoint '{}'".format(ckpt_file))

        if osp.isfile(ckpt_file):
            print("=> loading checkpoint '{}'".format(ckpt_file))
            checkpoint = torch.load(ckpt_file)
            start_epoch = checkpoint['epoch']

            self.best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))

        return start_epoch


