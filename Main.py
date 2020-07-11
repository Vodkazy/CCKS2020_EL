#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午11:04
  @ Author   : Vodka
  @ File     : Main .py
  @ Software : PyCharm
"""
from Eval import Eval
from config import *
from data_util import *

if __name__ == '__main__':
    set_random_seed(20200711)
    preprocess_pickle_file()  # 生成全局变量索引pickle文件
    preprocess_tsv_file()  # 生成模型训练、验证、推断所需的tsv文件
    generate_feature_pickle()
    # train_data = pd.read_csv(TSV_PATH + 'ET_TRAIN.tsv', sep='\t')
    # print(train_data.head())
    # 训练linking模型
    train_entity_linking_model('EL_BASE_EPOCH0.ckpt')
    generate_link_tsv_result('EL_BASE_EPOCH0.ckpt')
    # 训练typing模型
    train_entity_typing_model('ET_BASE_EPOCH1.ckpt')
    generate_type_tsv_result('ET_BASE_EPOCH1.ckpt')
    # 打印测试实例
    # el_ret = pd.read_csv("./data/result/ET_TEST_RESULT.tsv", sep='\t')
    # print(el_ret.head())
    #
    # path = "./data/result/ET_VALID_RESULT.tsv"
    # data = pd.read_csv(
    #     path, sep='\t', dtype={
    #         'text_id': np.str_,
    #         'offset': np.str_,
    #         'kb_id': np.str_
    #     })
    # print(data.head())
    # 验证
    make_predication_result('dev.json', 'valid_result.json', 'EL_VALID_RESULT.tsv', 'ET_VALID_RESULT.tsv')
    # 测试
    make_predication_result('test.json', 'test_result.json', 'EL_TEST_RESULT.tsv', 'ET_TEST_RESULT.tsv')
    # 评估
    eval = Eval('./data/ccks2020_el_data_v1/dev.json', './data/result/valid_result.json')

    prec, recall, f1 = eval.micro_f1()
    print(prec, recall, f1)
    if eval.errno:
        print(eval.errno)
