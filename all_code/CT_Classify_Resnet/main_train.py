# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import shutil
import time
import numpy as np
import pandas as pd
from logging import getLogger
import torch
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import (
    dataset_split,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter
)
from src.dataloader import TrainLoader, TestLoader
from src.resnet_3d import generate_model

logger = getLogger()

def parse_arguments():
    parser = argparse.ArgumentParser()

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="CT_images",
                        help="path to dataset repository")
    parser.add_argument('--target_file', type=str, default='/media/wukai/github_clone/CT_Classify_Resnet/all_label.csv',
                        help="path to dataset repository")
    parser.add_argument('--task', type=str, default='stage',
                        help="stage/grade")                   
    parser.add_argument("--kfolds", type=int, default=5,
                        help="Total folds for training and validating data set split")
    parser.add_argument("--fold_train", type=int, default=0,
                        help="fold for training")
    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=1000, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=0.001, type=float, help="base learning rate")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--optimizer", default="SGD", type=str, help="optimizer type")
    parser.add_argument("--loss_fn", default='ENTROPY', type=str, help="define the loss function")
    parser.add_argument("--rank", default=0, type=int, help="rank for distributed training")
    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--model_depth", default=18, type=int, help="resnet model depth: [10, 18, 34, 50, 101, 152, 200]")
    parser.add_argument("--hidden_mlp", default=512, type=int,
                        help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=2, type=int,
                        help="feature dimension")       
    parser.add_argument("--checkpoint_freq", type=int, default=10,
                        help="Save the model periodically")                       
    parser.add_argument("--use_fp16", type=bool, default=True,
                        help="whether to train with mixed precision or not")
    parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
    parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                        https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")
    parser.add_argument("--checkpoints_path", type=str, default="checkpoints",
                        help="checkpoints path for checkpoints and log")
                        
    parser.add_argument("--seed", type=int, default=1990, help="seed")
    parser.add_argument("--gpu_id", type=int, default=2, help="")
    return parser.parse_args()

def main():
    args = parse_arguments()
    torch.cuda.set_device(args.gpu_id)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # data set split
    args.training_cases, args.testing_cases = dataset_split(args) 

    # build data
    train_dataset = TrainLoader(args)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        #num_workers=args.batch_size,
        pin_memory=True,
        drop_last=True
    )
    test_dataset = TestLoader(args)
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        shuffle = False,
        batch_size=args.batch_size
    )
    logger.info("Train Task-%s Start."%args.task)

    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = generate_model(
        model_depth = args.model_depth,
        normalize=True,
        hidden_dim = 64,
        output_dim=args.feat_dim,

    )

    # synchronize batch norm layers
    '''
    if args.sync_bn == "pytorch":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    '''
    # copy model to GPU
    model = model.cuda()
    logger.info(model)
    logger.info("Building model done.")
    logger.info("For Tumor %s Prediction"%args.task)
    # Init scaler 
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    # build optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.wd,
        )
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            betas = (0.9,0.999),
            #weight_decay=args.wd,
        )

    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # build loss function
    if args.loss_fn == 'ENTROPY':
        loss_fn = CrossEntropyLoss()

    logger.info("Building optimizer done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        scaler = scaler
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        logger.info("============ Starting epoch %i ... ============" % epoch)
        # train the network
        losses = train(train_loader, model, optimizer, loss_fn, epoch, scaler, args)
        ACC = test(test_loader, model, test_dataset, args)
        training_stats.update(losses)
        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'scaler':scaler.state_dict()
            }
            if args.use_fp16:
                save_dict["amp"] = scaler.state_dict()
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.checkpoints_path, "ckp-" + str(epoch) + "_%s_fold.pth"%args.task),
                )
        scheduler.step()

def train(train_loader, model, optimizer, loss_fn, epoch, scaler, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for it, [image, target] in enumerate(train_loader):
        with torch.cuda.amp.autocast(enabled=False):
            # measure data loading time
            output = model(image.cuda())
            loss = loss_fn(F.softmax(output,dim=1), target.cuda())
        
        # ============ backward and optim step ... ============
        scaler.scale(loss).backward(retain_graph=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        print('Epoch:%d Batch:%d/%d loss:%f'%(epoch,it,len(train_loader),loss))
        losses.update(loss.item(), image.size(0))
    # ============ misc ... ============
    batch_time = (time.time() - end)
    if args.rank ==0:
        logger.info((
            "Epoch: %d\t"
            "Time %.2f\t"
            "Loss %.4f\t"
             )%(epoch,batch_time,losses.avg))
    return losses.avg

def test(test_loader, model, test_dataset, args):
    model.eval()
    end = time.time()

    images_id = [os.path.basename(image_dir).split('_')[0].replace('.npz','') for image_dir in test_dataset.images_list]
    end = time.time()
    for it, [image, target] in enumerate(test_loader):
        if it == 0:
            output = model(image.cuda()).detach().cpu()
            predict = F.softmax(output,dim=1).numpy()
            truth = target
        else:
            output = model(image.cuda()).detach().cpu()
            predict = np.vstack((predict, F.softmax(output,dim=1).numpy()))
            truth = np.concatenate((truth, target))

    AUC = roc_auc_score(y_true=truth,y_score = predict[:,1])
    # ============ misc ... ============
    if args.rank ==0:
        logger.info((
            "Test set\t"
            "Time %.2f\t"
            "AUC %.4f\t"
             )%(time.time()-end,AUC))


if __name__ == "__main__":
    main()
