# coding:utf-8

import argparse
import sys
import os
from data_manager import Data_Processor
from dual_attn import d_attn
import numpy as np

parser = argparse.ArgumentParser()
# Option for pre-processing data
parser.add_argument("-c", "--do_preprocess", type=bool,
                    help="True or False to preprocess raw data for ConvMF (default = False)", default=False)

parser.add_argument("-r", "--raw_rating_data_path", type=str,
                    help="Path to raw rating data. data format - user id::item id::rating"
                    , default="/Users/lys/Code/data/RS/yelp/yelp_training_set/")
parser.add_argument("-t", "--split_ratio", type=float,
                    help="Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and "
                         "test set, respectively (default = 0.2)", default=0.2)

# 切分后的数据地址
parser.add_argument("-d", "--data_path", type=str,
                    help="Path to training, valid and test data sets"
                    , default="processed/")
parser.add_argument("-a", "--aux_path", type=str, help="Path to R, D_all sets"
                    , default="processed/")

parser.add_argument("-m", "--input_size", type=str, help="Input vector length"
                    , default=10000)
parser.add_argument("-s", "--vocab_size", type=int,
                    help="Size of vocabulary (default = 8000)", default=8000)

args = parser.parse_args()
do_preprocess = args.do_preprocess

data_path = args.data_path
aux_path = args.aux_path

data_processor = Data_Processor()

if do_preprocess:
    raw_data_path = args.raw_rating_data_path
    split_ratio = args.split_ratio
    data_path += str(split_ratio)
    input_size=args.input_size

    R, D_all = data_processor.preprocess(raw_data_path,input_size=input_size)

    D_tvt=data_processor.generate_train_valid_test_file_from_R(
        data_path, R,D_all, split_ratio)
    data_processor.save(aux_path, R, D_all,D_tvt)
    print(R.shape)

else:
    split_ratio = args.split_ratio
    data_path += str(split_ratio)
    input_size=args.input_size
    vocab_size=args.vocab_size

    R, D_all,D_tvt = data_processor.load(aux_path)
    # vocab_size = len(D_all['X_vocab']) + 1



    d_attn(train_data=D_tvt['X_train'],valid_data=D_tvt['X_valid'],test_data=D_tvt['X_test'],D_all=D_all)

