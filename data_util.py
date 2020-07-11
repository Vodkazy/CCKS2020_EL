#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:40
  @ Author   : Vodka
  @ File     : data_util .py
  @ Software : PyCharm
"""

from DataFramePreprocessor import DataFramePreprocessor
from EntityLinkingModel import EntityLinkingModel
from EntityLinkingPredictor import EntityLinkingPredictor
from EntityLinkingProcessor import EntityLinkingProcessor
from EntityTypingModel import EntityTypingModel
from EntityTypingPredictor import EntityTypingPredictor
from EntityTypingProcessor import EntityTypingProcessor
from PicklePreprocessor import PicklePreprocessor
from config import *


def preprocess_pickle_file():
    print("preprocess_pickle_file .......")
    processor = PicklePreprocessor()
    processor.run()
    read_pickle()


def preprocess_tsv_file():
    print("preprocess_tsv_file .......")
    processor = DataFramePreprocessor()
    processor.run()


def generate_feature_pickle():
    print("generate_feature_pickle .......")
    processor = EntityLinkingProcessor()
    processor.generate_feature_pickle(max_length=384)

    processor = EntityTypingProcessor()
    processor.generate_feature_pickle(max_length=64)


def train_entity_linking_model(ckpt_name):
    print("train_entity_linking_model .......")
    model = EntityLinkingModel(max_length=384, batch_size=16)
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1,
        distributed_backend='dp',
        default_root_dir=EL_SAVE_PATH,
        profiler=True,
    )
    trainer.fit(model)
    trainer.save_checkpoint(CKPT_PATH + ckpt_name)


def train_entity_typing_model(ckpt_name):
    print("train_entity_typing_model .......")
    model = EntityTypingModel(max_length=64, batch_size=16)
    trainer = pl.Trainer(
        max_epochs=5,
        gpus=1,
        distributed_backend='dp',
        default_root_dir=ET_SAVE_PATH,
        profiler=True,
    )
    trainer.fit(model)
    trainer.save_checkpoint(CKPT_PATH + ckpt_name)


def generate_link_tsv_result(ckpt_name):
    print("generate_link_tsv_result .......")
    predictor = EntityLinkingPredictor(ckpt_name, batch_size=16, use_pickle=True)
    predictor.generate_tsv_result('EL_VALID.tsv', tsv_type='Valid')
    predictor.generate_tsv_result('EL_TEST.tsv', tsv_type='Test')


def generate_type_tsv_result(ckpt_name):
    print("generate_type_tsv_result .......")
    predictor = EntityTypingPredictor(ckpt_name, batch_size=16, use_pickle=True)
    predictor.generate_tsv_result('ET_VALID.tsv', tsv_type='Valid')
    predictor.generate_tsv_result('ET_TEST.tsv', tsv_type='Test')


def make_predication_result(input_name, output_name, el_ret_name, et_ret_name):
    print("make_predication_result .......")
    entity_to_kbids = PICKLE_DATA['ENTITY_TO_KBIDS']

    el_ret = pd.read_csv(
        RESULT_PATH + el_ret_name, sep='\t', dtype={
            'text_id': np.str_,
            'offset': np.str_,
            'kb_id': np.str_
        })

    et_ret = pd.read_csv(
        RESULT_PATH + et_ret_name, sep='\t', dtype={
            'text_id': np.str_,
            'offset': np.str_
        })

    result = []
    with open(RAW_PATH + input_name, 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            line = json.loads(line)
            for data in line['mention_data']:
                text_id = line['text_id']
                offset = data['offset']

                candidate_data = el_ret[(el_ret['text_id'] == text_id) & (el_ret['offset'] == offset)]
                # Entity Linking
                if len(candidate_data) > 0 and candidate_data['logits'].max() > 0:
                    max_idx = candidate_data['logits'].idxmax()
                    data['kb_id'] = candidate_data.loc[max_idx]['kb_id']
                # Entity Typing
                else:
                    type_data = et_ret[(et_ret['text_id'] == text_id) & (et_ret['offset'] == offset)]
                    data['kb_id'] = 'NIL_' + type_data.iloc[0]['result']
            result.append(line)

    with open(RESULT_PATH + output_name, 'w', encoding="utf-8") as f:
        for r in result:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')