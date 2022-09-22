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
import glob
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


def parse_arguments():
    parser = argparse.ArgumentParser()

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--data_path", type=str, default="CT_images",
                        help="path to dataset repository")
    parser.add_argument('--target_file', type=str, default='/share/Data01/wukai/CT_Classify_Resnet/CT_images/all_label.csv',
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
    parser.add_argument("--batch_size", default=16, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--base_lr", default=0.0001, type=float, help="base learning rate")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--optimizer", default="ADAM", type=str, help="optimizer type")
    parser.add_argument("--loss_fn", default='ENTROPY', type=str, help="define the loss function")
    parser.add_argument("--rank", default=0, type=int, help="rank for distributed training")
    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--model_depth", default=18, type=int, help="resnet model depth: [10, 18, 34, 50, 101, 152, 200]")
    parser.add_argument("--hidden_mlp", default=512, type=int,
                        help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=1, type=int,
                        help="feature dimension")       
    parser.add_argument("--checkpoint_freq", type=int, default=25,
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

    # data set split
    args.training_cases, args.testing_cases = dataset_split(args) 

    test_dataset = TestLoader(args)
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        shuffle = False,
        batch_size=args.batch_size
    )

    # build model
    model = generate_model(
        model_depth = args.model_depth,
        normalize=True,
        hidden_dim = 64,
        output_dim=args.feat_dim,
    )


    # copy model to GPU
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join('/share/Data01/wukai/CT_Classify_Resnet/checkpoints','ckp-10_stage_fold.pth'))['state_dict'])


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

    datatable = test(test_loader, model, test_dataset, args)
    datatable.to_csv('%s_test_prob.csv'%args.task,index=0)



def test(test_loader, model, test_dataset, args):
    model.eval()
    images_id = [os.path.basename(image_dir).split('_')[0].replace('.npz','') for image_dir in test_dataset.images_list]

    cases_id = []
    truth = []
    prob = []

    for it, image_dir in enumerate(test_dataset.images_list):
        case_id = os.path.basename(image_dir).split('_')[0].replace('.npz','')
        image = np.expand_dims(np.load(image_dir)['data'],0)
        image = torch.from_numpy(image).unsqueeze(dim = 0)
        output = model(image.cuda()).detach().cpu()
        predict = torch.sigmoid(output).numpy().squeeze().item()

        cases_id.append(case_id)
        truth.append(test_dataset.target_dict.get(case_id))
        prob.append(predict)

    datatable = {'case_id':cases_id, 'truth':truth, 'prob':prob}

    AUC = roc_auc_score(y_true=truth,y_score = prob)
    return pd.DataFrame(datatable)


if __name__ == "__main__":
    main()
