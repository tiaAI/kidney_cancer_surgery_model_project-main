# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import dataset
from utils import (
    dataset_split,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter
)
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

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=1000, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument("--model_depth", default=18, type=int, help="resnet model depth: [10, 18, 34, 50, 101, 152, 200]")
    parser.add_argument("--hidden_mlp", default=512, type=int,
                        help="hidden layer dimension in projection head")
    parser.add_argument("--feat_dim", default=1, type=int,
                        help="feature dimension")       
                
    parser.add_argument("--checkpoints_path", type=str, default="checkpoints",
                        help="checkpoints path for checkpoints and log")
                        
    parser.add_argument("--seed", type=int, default=1990, help="seed")
    parser.add_argument("--gpu_id", type=int, default=2, help="")
    return parser.parse_args()

def main():
    args = parse_arguments()
    torch.cuda.set_device(args.gpu_id)
    fix_random_seeds(args.seed)

    # build data
    files_dir = glob.glob(os.path.join(args.data_path,'cropped/*.npz'))
    input_images = []
    case_ids = []
    for file_dir in files_dir:
        input_images.append(np.expand_dims(np.load(file_dir)['data'],axis=0))
        case_id = os.path.basename(file_dir).replace('.npz','')
        case_ids.append(case_id)

    # build model
    model = generate_model(
        model_depth = args.model_depth,
        normalize=True,
        hidden_dim = 64,
        output_dim=args.feat_dim,

    )

    # copy model to GPU
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join('/share/Data01/wukai/CT_Classify_Resnet/checkpoints','ckp-40_grade_fold.pth'))['state_dict'])

    features_table = features_extract(model, input_images,case_ids, args)
    features_table.to_csv('/share/Data01/wukai/CT_Classify_Resnet/grade_pred_features.csv',index=0,sep=',')


def features_extract(model, input_images,case_ids, args):
    feature_table = pd.DataFrame(columns=['case_id']+['feat%d'%(i) for i in range(64)])
    model.eval()
    for it, [image, case_id] in enumerate(zip(input_images,case_ids)):
        # measure data loading time
        print("process %s"%(case_id))
        input = torch.from_numpy(image).unsqueeze(dim = 0)
        x,feat = model(input.cuda())
        features = [case_id] + [str(f) for f in list(feat.squeeze().data.cpu().numpy())]
        feature_table.loc[it] = features
    return feature_table

if __name__ == "__main__":
    main()
