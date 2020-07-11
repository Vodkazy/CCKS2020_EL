#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:43
  @ Author   : Vodka
  @ File     : DataFramePreprocessor .py
  @ Software : PyCharm
"""
from config import *

logger = logging.getLogger(__name__)


class DataFramePreprocessor:
    """生成模型训练、验证、推断所需的tsv文件"""

    def __init__(self):
        pass

    def process_link_data(self, input_path, output_path, max_negs=-1):

        entity_to_kbids = PICKLE_DATA['ENTITY_TO_KBIDS']
        # print("entity_to_kbids")
        kbid_to_text = PICKLE_DATA['KBID_TO_TEXT']
        # print(kbid_to_text)
        kbid_to_predicates = PICKLE_DATA['KBID_TO_PREDICATES']
        link_dict = defaultdict(list)

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)

                for data in line['mention_data']:
                    # 对测试集特殊处理
                    if 'kb_id' not in data:
                        data['kb_id'] = '0'

                    # KB中不存在的实体不进行链接（NIL）
                    if not data['kb_id'].isdigit():
                        continue

                    entity = data['mention']
                    kbids = list(entity_to_kbids[entity])
                    random.shuffle(kbids)

                    num_negs = 0
                    for kbid in kbids:
                        # if num_negs >= max_negs and kbid != data['kb_id']:
                        #     continue

                        link_dict['text_id'].append(line['text_id'])
                        link_dict['entity'].append(entity)
                        link_dict['offset'].append(data['offset'])
                        link_dict['short_text'].append(line['text'])
                        link_dict['kb_id'].append(kbid)
                        link_dict['kb_text'].append(kbid_to_text[kbid])
                        link_dict['kb_predicate_num'].append(len(kbid_to_predicates[kbid]))
                        # 通过entity_mention -> candidate entity set的映射 生成正例和反例
                        if kbid != data['kb_id']:
                            link_dict['predict'].append(0)
                            num_negs += 1
                        else:
                            link_dict['predict'].append(1)

        link_data = pd.DataFrame(link_dict)
        link_data.to_csv(output_path, index=False, sep='\t')

    def process_type_data(self, input_path, output_path):
        kbid_to_types = PICKLE_DATA['KBID_TO_TYPES']
        type_dict = defaultdict(list)

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)

                for data in line['mention_data']:
                    entity = data['mention']

                    # 测试集特殊处理
                    if 'kb_id' not in data:
                        entity_type = ['Other']
                    elif data['kb_id'].isdigit():
                        entity_type = kbid_to_types[data['kb_id']]
                    else:
                        entity_type = data['kb_id'].split('|')  # NIL_Organization|NIL_Location
                        for x in range(len(entity_type)):
                            entity_type[x] = entity_type[x][4:]
                    for e in entity_type:
                        type_dict['text_id'].append(line['text_id'])
                        type_dict['entity'].append(entity)
                        type_dict['offset'].append(data['offset'])
                        type_dict['short_text'].append(line['text'])
                        type_dict['type'].append(e)

        type_data = pd.DataFrame(type_dict)
        type_data.to_csv(output_path, index=False, sep='\t')

    def run(self):
        self.process_link_data(
            input_path=RAW_PATH + 'train.json',
            output_path=TSV_PATH + 'EL_TRAIN.tsv',
            max_negs=2,
        )
        logger.info('Process EL_TRAIN Finish.')
        self.process_link_data(
            input_path=RAW_PATH + 'dev.json',
            output_path=TSV_PATH + 'EL_VALID.tsv',
            max_negs=-1,
        )
        logger.info('Process EL_VALID Finish.')
        self.process_link_data(
            input_path=RAW_PATH + 'test.json',
            output_path=TSV_PATH + 'EL_TEST.tsv',
            max_negs=-1,
        )
        logger.info('Process EL_TEST Finish.')

        self.process_type_data(
            input_path=RAW_PATH + 'train.json',
            output_path=TSV_PATH + 'ET_TRAIN.tsv',
        )
        logger.info('Process ET_TRAIN Finish.')
        self.process_type_data(
            input_path=RAW_PATH + 'dev.json',
            output_path=TSV_PATH + 'ET_VALID.tsv',
        )
        logger.info('Process ET_VALID Finish.')
        self.process_type_data(
            input_path=RAW_PATH + 'test.json',
            output_path=TSV_PATH + 'ET_TEST.tsv',
        )
        logger.info('Process ET_TEST Finish.')
