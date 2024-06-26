from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime
import wandb

# Oof
import eval as eval_script

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
parser.add_argument('--refinement_mode', default=False, required=False, type=bool, help='for the refinement training of real data on a pretrained pbr model')
parser.add_argument('--refinement_iterations', default=1000, required=False, type=int, help='iterations for refinement')
parser.add_argument('--start_epoch', default=0, required=False, type=int)

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I']

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):
        preds = self.net(images)
        losses = self.criterion(self.net, preds, targets, masks, num_crowds)
        return losses

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out
    
def refinement_training():
    DEBUG = False
    print("entering refinement mode...")
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))
    
    real_dataset = COCODetection(image_path = cfg.dataset.train_images,
                                        info_file=cfg.dataset.train_info,
                                        transform=SSDAugmentation(MEANS))
    
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()
    #print("Number of iterations: " + str(iteration))
    epoch_size = len(real_dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    #print("size per epoch: " + str(epoch_size))
    #print("number of epochs: " + str(num_epochs))

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    pbr_data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True,
                                  generator=torch.Generator(device='cuda'))
    pbr_iterator = iter(pbr_data_loader)

    
    real_data_loader = data.DataLoader(real_dataset, args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=True, collate_fn=detection_collate,
                                        pin_memory=True,
                                        generator=torch.Generator(device='cuda'))
    real_iterator = iter(real_data_loader)
    
    # need a second data loader for the validation set
    # note that the val_dataset uses BaseTransform as transformation, not SSDAugmentation like during training
    # this is because during training the SSDAugmentation also randomly flips,etc... images for robustness
    val_data_loader = data.DataLoader(val_dataset, args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False, collate_fn=detection_collate,
                                      pin_memory=True,
                                      generator=torch.Generator(device='cuda'))
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    #create a dict for the val_loss
    #val_loss_avgs = { k: MovingAverage(100) for k in loss_types}

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    pbr_samples = 0
    real_samples = 0
    try:
        #for epoch in range(num_epochs):
        epoch=args.start_epoch
        while(True):
        
            # Resume from start_iter
            #if (epoch+1)*epoch_size < iteration:
            #    continue

            #print("DEBUGGING VAL LOSS")
            #compute_validation_loss(net, val_data_loader, log, epoch)
            
            #for datum in real_data_loader:
            #    if False:
            if np.random.randint(0, high=100) < (100 * cfg.ratio_pbr_to_real): 
                pbr_samples += 1
                print("Sampling PBR")
                try:
                    datum = next(pbr_iterator)
                except StopIteration: # incase the dataloader is exhausted
                    print("PBR iterator is exhausted")
                    pbr_iterator = iter(pbr_data_loader)
                    datum = next(pbr_iterator)
            else:
                real_samples += 1
                print("Sampling Real")
                try:
                    datum = next(real_iterator)
                except StopIteration: #again incase the dataloader is exhausted
                    print("Real iterator is exhausted")
                    real_iterator = iter(real_data_loader)
                    datum = next(real_data_loader)
            
        
            
            # Stop if we've reached an epoch if we're resuming from start_iter
            #if iteration == (epoch+1)*epoch_size:
            #    break

            # Stop at the configured number of iterations even if mid-epoch
            if iteration == cfg.max_iter:
                break
            
            #for augmentation debug
            #exit() 
            
            # Change a config setting if we've reached the specified iteration
            changed = False
            for change in cfg.delayed_settings:
                if iteration >= change[0]:
                    changed = True
                    cfg.replace(change[1])

                    # Reset the loss averages because things might have changed
                    for avg in loss_avgs:
                        avg.reset()
            
            # If a config setting was changed, remove it from the list so we don't keep checking
            if changed:
                cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

            # Warm up by linearly interpolating the learning rate from some smaller value
            if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

            # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
            while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                step_index += 1
                set_lr(optimizer, args.lr * (args.gamma ** step_index))
            
            if (iteration % 100 == 0):
                #print("Iteration=" + str(iteration) + " | Learning-rates check:")
                for i,param_group in enumerate(optimizer.param_groups):
                    wandb.log({'lr-param_group_' + str(i) : param_group['lr']}, step=iteration)
                    print(param_group['lr'])
                #print("Done printing learning rates!")

            # Zero the grad to get ready to compute gradients
            optimizer.zero_grad()

            # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
            try:
                losses = net(datum)
            except IndexError as e:
                print("An index error occured!")
                #print("Datum:" + str(datum))
                loss.backward() 
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                #print("Datum:" + str(datum))
                if DEBUG:
                    exit()
                loss.backward() 
                continue
            
            losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
            loss = sum([losses[k] for k in losses])
            
            #avg loss for wandb?
            #avg_loss += loss.item()
            

            # no_inf_mean removes some components from the loss, so make sure to backward through all of it
            # all_loss = sum([v.mean() for v in losses.values()])

            # Backprop
            loss.backward() # Do this to free up vram even if loss is not finite
            if torch.isfinite(loss).item():
                optimizer.step()
            
            # Add the loss to the moving average for bookkeeping
            for k in losses:
                loss_avgs[k].add(losses[k].item())

            cur_time  = time.time()
            elapsed   = cur_time - last_time
            last_time = cur_time

            # Exclude graph setup from the timing information
            if iteration != args.start_iter:
                time_avg.add(elapsed)

            if iteration % 10 == 0:
                eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                
                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                if WANDB:
                    loss_dict = {}
                    for k in losses:
                        loss_dict['moving_avg_loss_' + k] = round(loss_avgs[k].get_avg(),5)
                    wandb.log(loss_dict, step=iteration)
                print(f"Pbr-Samples: {pbr_samples} | Real-Samples: {real_samples}")
                print(('Epoch: %3d | Iteration %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                        % tuple([epoch,iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                

            if args.log:
                precision = 5
                loss_info = {k: round(losses[k].item(), precision) for k in losses}
                loss_info['T'] = round(loss.item(), precision)
                if WANDB:
                    loss_dict = {}
                    loss_dict = {'loss_'+k: round(losses[k].item(),precision) for k in losses}
                    loss_dict['loss_T'] = round(loss.item(), precision)
                    wandb.log(loss_dict, step=iteration)
                
                
                if args.log_gpu:
                    log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                    
                log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                    lr=round(cur_lr, 10), elapsed=elapsed)

                log.log_gpu_stats = args.log_gpu
            
            #also compute validation loss every 3000 iterations, NOT NECESSARY FOR REFINEMENT
            if iteration > 0 and iteration % 3000 == 0:
                compute_validation_loss(net, val_data_loader, log, epoch, iteration)

            iteration += 1

            if iteration % args.save_interval == 0 and iteration != args.start_iter:
                if args.keep_latest:
                    latest = SavePath.get_latest(args.save_folder, cfg.name)

                print('Saving state, iter:', iteration)
                yolact_net.save_weights(save_path(epoch, iteration))

                if args.keep_latest and latest is not None:
                    if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                        print('Deleting old save...')
                        os.remove(latest)


            #if (iteration % 500) == 0:
            #    print("Sampling ratio so far: Pbr=" + str(pbr_samples) + " | Real=" + str(real_samples))
            # This is done per epoch
            #if args.validation_epoch > 0:
            #    if epoch % args.validation_epoch == 0 and epoch > 0:
            #        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        
        print("Training is done! Now computing validation mAP a final time...")
        
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()
    print("Sampled Pbr: " + str(pbr_samples))
    print("Samples Real: " + str(real_samples))
    yolact_net.save_weights(save_path(epoch, iteration))
    return

def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS))
    
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:
        net = net.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    print("size per epoch: " + str(epoch_size))
    print("number of epochs: " + str(num_epochs))

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True,
                                  generator=torch.Generator(device='cuda'))
    
    # need a second data loader for the validation set
    # note that the val_dataset uses BaseTransform as transformation, not SSDAugmentation like during training
    # this is because during training the SSDAugmentation also randomly flips,etc... images for robustness
    val_data_loader = data.DataLoader(val_dataset, args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False, collate_fn=detection_collate,
                                      pin_memory=True,
                                      generator=torch.Generator(device='cuda'))
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    #create a dict for the val_loss
    #val_loss_avgs = { k: MovingAverage(100) for k in loss_types}

    print('Begin training!')
    print()
    if args.refinement_mode:
        refinement_training()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):

            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue

            #print("DEBUGGING VAL LOSS")
            #compute_validation_loss(net, val_data_loader, log, epoch)
            
            for datum in data_loader:

                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break
                
                #for augmentation debug
                #exit() 
                
                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)
                
                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])
                
                #avg loss for wandb?
                #avg_loss += loss.item()
                

                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    if WANDB:
                        loss_dict = {}
                        for k in losses:
                            loss_dict['moving_avg_loss_' + k] = round(loss_avgs[k].get_avg(),5)
                        wandb.log(loss_dict, step=iteration)
                    
                    print(('Epoch: %3d | Iteration %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)
                    

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)
                    if WANDB:
                        loss_dict = {}
                        loss_dict = {'loss_'+k: round(losses[k].item(),precision) for k in losses}
                        loss_dict['loss_T'] = round(loss.item(), precision)
                        wandb.log(loss_dict, step=iteration)
                    
                    
                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                
                #also compute validation loss every 3000 iterations 
                if iteration > 0 and iteration % 3000 == 0:
                    compute_validation_loss(net, val_data_loader, log, epoch, iteration)

                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)


            
            # This is done per epoch
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        
        print("Training is done! Now computing validation mAP a final time...")
        # one final validation loss calculation:
        compute_validation_loss(net, val_data_loader, log, epoch, iteration)
        
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_folder)
            
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))

"""
def compute_validation_loss(net, dataset, log : Log):
    #Calculates the loss on the validation dataset.
    print('Calculating validaton losses, this may take a while...')
    global loss_types
    with torch.no_grad():
        net.eval()
        losses = {}
        dataset_indices = list(range(len(dataset)))
        dataset_indices = dataset_indices[:100]
        # Don't switch to eval mode here. Warning: this is viable but changes the interpretation of the validation loss.
        # trial with and without eval()
        for it, image_idx in enumerate(dataset_indices):
            img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)
            continue
            batch = Variable(img.unsqueeze(0))
            preds = net(batch)
        
        #loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        #print(('Validation Loss||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)
        net.train()
"""

def compute_validation_loss(net, data_loader, log : Log, epoch, i):
    #Calculates the loss on the validation dataset.
    # note: epoch and iter are from the training loop, not the local epoch and iter
    print('Calculating validaton losses, this may take a while...')
    start_time = time.time()
    global loss_types

    with torch.no_grad():
        val_loss_avg = {}
        val_loss_avg['B'] = 0
        val_loss_avg['M'] = 0
        val_loss_avg['C'] = 0
        val_loss_avg['S'] = 0
        val_loss_avg['T'] = 0

        losses = {}
        problematic_datums = []
        #net.eval()
        # Don't switch to eval mode here. Warning: this is viable but changes the interpretation of the validation loss.
        # TODO: trial with and without eval()
        iterations = 0
        for datum in data_loader:
            iterations += 1
            #if i > 1000:
            #    break
            try:
                losses = net(datum) 
            except Exception as e:  
                print(e)
                problematic_datums.append(datum)
                continue
            losses = { k: (v).mean() for k,v in losses.items() }
            loss = sum([losses[k] for k in losses])
            
            precision = 5
            val_loss_info = {k: round(losses[k].item(), precision) for k in losses}
            val_loss_info['T'] = round(loss.item(), precision)
            
            val_loss_avg['B'] += val_loss_info['B']
            val_loss_avg['M'] += val_loss_info['M']
            val_loss_avg['C'] += val_loss_info['C']
            val_loss_avg['S'] += val_loss_info['S']
            val_loss_avg['T'] += val_loss_info['T']
        
        for key in val_loss_avg:
            val_loss_avg[key] = (val_loss_avg[key]/iterations)
        end_time = time.time()

        print(val_loss_avg)
        log.log('val-loss', val_loss=val_loss_avg, epoch=epoch, iter=i, elapsed=(end_time - start_time))
        
        if WANDB:
            loss_dict = {}
            for k in val_loss_avg:
                loss_dict['val_'+k] = val_loss_avg[k]
            wandb.log(loss_dict, step=i)
            
        #f = open('problematic_datums.txt', 'a')
        #f.write('\n\n------------------------------------------------')
        #f.write('\n\nNEW CALCULATION ITERATION \n\n')
        #f.write('------------------------------------------------')

        #f.write('Nr. Problematic Datums: ' + str(len(problematic_datums)) + '\n\n')
        #f.write(str(problematic_datums))
        #f.close()
        #net.train()
    

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()
    
#this is the old compute_validation loss class from dbolya
"""
def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        #wandb.log({"validation-loss" : loss_labels})
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)
"""



def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    WANDB = True
    try:
        if WANDB:
            if args.refinement_mode:
                print("entering refinement wandb")
                wandb.init(
                    project = 'BscThesis',

                    config= {
                        'config_name' : cfg.name,
                        'learning-rate' : cfg.lr,
                        'architecture' : args.config,
                        'dataset' : cfg.dataset,
                        'iterations' : cfg.max_iter, 
                        'refinement-mode' : True,
                        'ratio_pbr_to_real' : cfg.ratio_pbr_to_real
                    }
                )
            else:
                wandb.init(
                project = 'BscThesis',

                config= {
                    'config_name' : cfg.name,
                    'learning-rate' : cfg.lr,
                    'architecture' : args.config,
                    'dataset' : cfg.dataset,
                    'iterations' : cfg.max_iter, 
                    'refinement-mode' : False
                }
                )
        if args.refinement_mode:
            refinement_training()
        else:
            train()
    finally:    
        #finish wandb, unsure if this is actually necessary
        if WANDB:
            wandb.finish()
    """
    try:
        # initialize wandb
        WANDB = False
        if WANDB:
            wandb.init(
                project = 'BscThesis',

                config= {
                    'learning-rate' : cfg.lr,
                    'architecture' : 'yolact_resnet50_ssd_trial',
                    'dataset' : 'trial-subset',
                    'iterations' : cfg.max_iter, 
                }
            )
        
    except:
        print("ERROR!")
    else: 
        print("OK!")
    finally:    
        #finish wandb, unsure if this is actually necessary
        if WANDB:
            wandb.finish()
            """
