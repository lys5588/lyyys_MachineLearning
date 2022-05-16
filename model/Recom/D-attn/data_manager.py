import os
import sys
import pickle

import json

import numpy as np
import pandas as pd
# 使用此方法获取item对象指定域的值
from operator import itemgetter
# 稀疏矩阵压缩，存放三元组数据
from scipy.sparse.csr import csr_matrix

import random

import torch
from nltk import word_tokenize
from torch.utils import data
from torch.utils.data.dataloader import default_collate

# 文本向量化包
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class Data_Processor():
    def load(self, path):
        # rb read byte
        R = pickle.load(open(path + "/ratings.all", "rb"))
        print("Load preprocessed rating data - %s" % (path + "/ratings.all"))
        D_all = pickle.load(open(path + "/document.all", "rb"))
        print("Load preprocessed document data - %s" % (path + "/document.all"))
        D_tvt = pickle.load(open(path + "/trainvalidtest.all", "rb"))
        print("Load model document data - %s" % (path + "/trainvalidtest.all"))
        return R, D_all, D_tvt

    def save(self, path, R, D_all, D_tvt):
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving preprocessed rating data - %s" % (path + "/ratings.all"))
        pickle.dump(R, open(path + "/ratings.all", "wb"))
        print("Done!")
        print("Saving preprocessed document data - %s" % (path + "/document.all"))
        pickle.dump(D_all, open(path + "/document.all", "wb"))
        print("Done!")
        print("Saving model document data - %s" % (path + "/trainvalidtest.all"))
        pickle.dump(D_tvt, open(path + "/trainvalidtest.all", "wb"))
        print("Done!")

    def read_rating(self, path):
        results = []
        if os.path.isfile(path):
            raw_ratings = open(path, 'r')
        else:
            print("Path (preprocessed) is wrong!")
            sys.exit()
        index_list = []
        rating_list = []
        all_line = raw_ratings.read().splitlines()
        for line in all_line:
            tmp = line.split()
            num_rating = int(tmp[0])
            if num_rating > 0:
                tmp_i, tmp_r = zip(*(elem.split(":") for elem in tmp[1::]))
                index_list.append(np.array(tmp_i, dtype=int))
                rating_list.append(np.array(tmp_r, dtype=float))
            else:
                index_list.append(np.array([], dtype=int))
                rating_list.append(np.array([], dtype=float))

        results.append(index_list)
        results.append(rating_list)

        return results

    def split_data(self, ratio, R):
        print("Randomly splitting rating data into training set (%.1f) and test set (%.1f)..." % (1 - ratio, ratio))
        train = []
        for i in range(R.shape[0]):
            user_rating = R[i].nonzero()[1]
            np.random.shuffle(user_rating)
            train.append((i, user_rating[0]))

        # "*train" to open a list
        remain_item = set(range(R.shape[1])) - set(list(zip(*train))[1])

        # to make sure that training set contains at least a rating on every user and item
        for j in remain_item:
            item_rating = R.tocsc().T[j].nonzero()[1]
            np.random.shuffle(item_rating)
            train.append((item_rating[0], j))

        rating_list = set(zip(R.nonzero()[0], R.nonzero()[1]))
        total_size = len(rating_list)
        remain_rating_list = list(rating_list - set(train))
        random.shuffle(remain_rating_list)

        num_addition = int((1 - ratio) * total_size) - len(train)
        if num_addition < 0:
            print('this ratio cannot be handled')
            sys.exit()
        else:
            train.extend(remain_rating_list[:num_addition])
            tmp_test = remain_rating_list[num_addition:]
            random.shuffle(tmp_test)
            valid = tmp_test[::2]
            test = tmp_test[1::2]

            trainset_u_idx, trainset_i_idx = zip(*train)
            trainset_u_idx = set(trainset_u_idx)
            trainset_i_idx = set(trainset_i_idx)
            if len(trainset_u_idx) != R.shape[0] or len(trainset_i_idx) != R.shape[1]:
                print("Fatal error in split function. Check your data again or contact authors")
                sys.exit()

        print("Finish constructing training set and test set")
        return train, valid, test

    def generate_train_valid_test_file_from_R(self, path, R, D_all, ratio):
        '''
        Split randomly rating matrix into training set, valid set and test set with given ratio (valid+test)
        and save three data sets to given path.
        Note that the training set contains at least a rating on every user and item.

        Input:
        - path: path to save training set, valid set, test set
        - R: rating matrix (csr_matrix)
        - ratio: (1-ratio), ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively
        '''
        # 得到评分用户物品二元组序列切分结果
        train, valid, test = self.split_data(ratio, R)
        print("Save training set and test set to %s..." % path)
        if not os.path.exists(path):
            os.makedirs(path)

        R_lil = R.tolil()
        D_user = D_all['X_user']
        D_item = D_all['X_item']

        D_train=[]
        for i,j in train:
            D_train.append([i,j,R_lil[i,j]])
        D_valid = []
        for i, j in valid:
            D_valid.append([i, j, R_lil[i, j]])
        D_test = []
        for i, j in test:
            D_test.append([i, j, R_lil[i, j]])

        #原来的单独将所有二元组对应数据组成大矩阵的方法，内存消耗过大，已放弃
        # D_train = []
        # for i, j in train:
        #     D_user_i = D_user[i]
        #     D_item_j = D_item[j]
        #     rate_ij = R_lil[i, j]
        #     D_train.append([D_user_i, D_item_j, rate_ij])
        #
        # D_valid = []
        # for i, j, in valid:
        #     D_user_i = D_user[i]
        #     D_item_j = D_item[j]
        #     rate_ij = R_lil[i, j]
        #     D_valid.append([D_user_i, D_item_j, rate_ij])
        #
        # D_test = []
        # for i, j, in test:
        #     D_user_i = D_user[i]
        #     D_item_j = D_item[j]
        #     rate_ij = R_lil[i, j]
        #     D_test.append([D_user_i, D_item_j, rate_ij])
        #
        # print("Done!")
        #
        # D_tvt={
        #     'X_train': D_train,
        #     'X_valid': D_valid,
        #     'X_test': D_test,
        # }

        D_tvt={
            'X_train': D_train,
            'X_valid': D_valid,
            'X_test': D_test,
        }
        return D_tvt

    # 预处理数据
    # 1. 处理business文件，得到 bussness对应的index并记录
    # 2. 处理user 文件并得到user对应的index并记录
    # 3. 处理review文件，得到三个数据
    #     1. 用户评分 三个列表，user，item，rate
    #     2. 得到同个用户的所有评论文本，并在文本间放一个标识符
    #     3. 得到同个物品的所有评论文本，并在文本间放一个标识符
    def preprocess(self, file_path, _max_df=0.5, _vocab_size=8000,input_size=10000):
        '''
                Preprocess rating and document data.

                Input:
                    - path_rating: path for data，type:json including user,business,review file


                Output:
                    - R: rating matrix (csr_matrix: row - user, column - item)
                    - D_all['user_sequence']: list of sequence of word index of each user's review ([[1,2,3,4,..],[2,3,4,...],...])
                    - D_all['item_sequence']: list of sequence of word index of each item's review ([[1,2,3,4,..],[2,3,4,...],...])
                '''

        raw_user_file = file_path + "user.json"
        raw_business_file = file_path + "business.json"
        raw_review_file = file_path + "review.json"

        # Validate data paths
        if os.path.isfile(raw_user_file):
            raw_user = open(raw_user_file, 'r')
            print("Path - rating data: %s" % raw_user_file)
        else:
            print("Path(rating) is wrong!")
            sys.exit()

        if os.path.isfile(raw_business_file):
            raw_business = open(raw_business_file, 'r')
            print("Path - document data: %s" % raw_business_file)
        else:
            print("Path(item text) is wrong!")
            sys.exit()

        if os.path.isfile(raw_review_file):
            raw_content = open(raw_review_file, 'r')
            print("Path - document data: %s" % raw_review_file)
        else:
            print("Path(item text) is wrong!")
            sys.exit()

        # 处理原始数据
        # 1.处理business文件，得到 bussness对应的index并记录

        business_idx = 0
        business_ids = set()
        business_id2index = dict()
        for line in raw_business.readlines():
            tmp_json = json.loads(line)
            tmp_id = tmp_json["business_id"]
            if tmp_id not in business_ids:
                business_id2index[tmp_id] = business_idx
                business_ids.add(tmp_id)
                business_idx += 1
        raw_business.close()

        # 2.处理user 文件并得到user对应的index并记录
        user_idx = 0
        user_ids = set()
        user_id2index = dict()
        for line in raw_user.readlines():
            tmp_json = json.loads(line)
            tmp_id = tmp_json["user_id"]
            if tmp_id not in user_ids:
                user_id2index[tmp_id] = user_idx
                user_ids.add(tmp_id)
                user_idx += 1
        raw_user.close()

        # 3. 处理review文件，得到三个数据
        # 3.1 得到用户评分 三个列表，user，item，rate

        user = []
        item = []
        rating = []

        user_review_list = {}
        item_review_list = {}

        raw_X = []

        for line in raw_content.readlines():
            tmp_json = json.loads(line)
            # 先处理得到评价矩阵
            tmp_id = tmp_json["user_id"]
            if tmp_id not in user_ids:
                user_id2index[tmp_id] = user_idx
                user_ids.add(tmp_id)
                user_idx += 1
            u_idx = user_id2index[tmp_json["user_id"]]

            tmp_id = tmp_json["business_id"]
            if tmp_id not in business_ids:
                business_id2index[tmp_id] = business_idx
                business_ids.add(tmp_id)
                business_idx += 1
            i_idx = business_id2index[tmp_json["business_id"]]

            tmp_rate = tmp_json["stars"]
            user.append(u_idx)
            item.append(i_idx)
            rating.append(float(tmp_rate))

            # 处理得到不同用户的评论文本
            review_text = tmp_json["text"]
            user_review_list.setdefault(u_idx, [])
            user_review_list[u_idx].append(review_text)
            item_review_list.setdefault(i_idx, [])
            item_review_list[i_idx].append(review_text)

            raw_X.append(review_text)
        raw_content.close()

        R = csr_matrix((rating, (user, item)))

        # 构建词典
        vectorizer = TfidfVectorizer(max_df=_max_df, stop_words={
            'english'}, max_features=_vocab_size)
        vectorizer.fit(raw_X)

        vocab = vectorizer.vocabulary_
        X_vocab = sorted(vocab.items(), key=itemgetter(1))

        X_sequence_user = {}
        for tmp_user_id in user_ids:
            seq = []
            sentences = user_review_list[user_id2index[tmp_user_id]]
            for sentence in sentences:
                sen = [float(vocab[word]) + 1 for word in sentence.split() if word in vocab]
                seq += sen
                seq.append(8000)
            full_seq = np.full((input_size), 8000)
            for i in range(min(len(seq), input_size)):
                full_seq[i] = seq[i]
            X_sequence_user[user_id2index[tmp_user_id]] = full_seq

        X_sequence_item = {}
        for tmp_item_id in business_ids:
            seq = []
            sentences = item_review_list[business_id2index[tmp_item_id]]
            for sentence in sentences:
                sen = [float(vocab[word]) + 1 for word in sentence.split() if word in vocab]
                seq += sen
                seq.append(8000)
            full_seq = np.full((input_size), 8000)
            for i in range(min(len(seq),input_size)):
                full_seq[i]=seq[i]
            X_sequence_item[business_id2index[tmp_item_id]] = full_seq

        # 对评论文本进行拼接操作
        user_review = {}
        item_review = {}
        whole_text = []

        D_all = {
            'X_user': X_sequence_user,
            'X_item': X_sequence_item,
            'X_vocab': X_vocab,
        }

        return R, D_all
