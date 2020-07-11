#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:42
  @ Author   : Vodka
  @ File     : PicklePreprocessor .py
  @ Software : PyCharm
"""
from config import *

logger = logging.getLogger(__name__)


class PicklePreprocessor:
    """生成全局变量Pickle文件的预处理器"""

    def __init__(self):

        # 实体名称对应的KBID列表  {"张健"} -> "10001"
        self.entity_to_kbids = defaultdict(set)

        # KBID对应的实体名称列表 "10001" -> {"张健"}
        self.kbid_to_entities = dict()

        # KBID对应的属性文本  "10001" -> {"政治面貌:中共党员","义项描述:潜山县塔畈乡副主任科员、纪委副书记","性别:男",
        # "学历:大专","中文名:张健"}
        self.kbid_to_text = dict()

        # KBID对应的实体类型列表 "10001" -> {"Person"}
        self.kbid_to_types = dict()

        # KBID对应的属性列表 "10001" -> {"政治面貌","义项描述","性别","学历","中文名"}
        self.kbid_to_predicates = dict()

        # 索引类型映射列表 ["Person"]
        self.idx_to_type = list()

        # 类型索引映射字典 {"Person":0}
        self.type_to_idx = dict()

    def run(self, shuffle_text=True):
        with open(RAW_PATH + 'kb.json', 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = json.loads(line)

                kbid = line['subject_id']
                # 将实体名与别名合并
                entities = set(line['alias'])
                entities.add(line['subject'])
                for entity in entities:
                    self.entity_to_kbids[entity].add(kbid)
                self.kbid_to_entities[kbid] = entities

                text_list, predicate_list = [], []
                for x in line['data']:
                    # 简单拼接predicate与object，这部分可以考虑别的方法尝试
                    text_list.append(':'.join([x['predicate'].strip(), x['object'].strip()]))
                    predicate_list.append(x['predicate'].strip())
                if shuffle_text:  # 对属性文本随机打乱顺序
                    random.shuffle(text_list)
                self.kbid_to_predicates[kbid] = predicate_list
                self.kbid_to_text[kbid] = ' '.join(text_list)

                # 删除文本中的特殊字符
                for c in ['\r', '\t', '\n']:
                    self.kbid_to_text[kbid] = self.kbid_to_text[kbid].replace(c, '')

                type_list = line['type'].split('|')
                self.kbid_to_types[kbid] = type_list
                for t in type_list:
                    if t not in self.type_to_idx:
                        self.type_to_idx[t] = len(self.idx_to_type)
                        self.idx_to_type.append(t)

        # 保存pickle文件
        pd.to_pickle(self.entity_to_kbids, PICKLE_PATH + 'ENTITY_TO_KBIDS.pkl')
        pd.to_pickle(self.kbid_to_entities, PICKLE_PATH + 'KBID_TO_ENTITIES.pkl')
        pd.to_pickle(self.kbid_to_text, PICKLE_PATH + 'KBID_TO_TEXT.pkl')
        pd.to_pickle(self.kbid_to_types, PICKLE_PATH + 'KBID_TO_TYPES.pkl')
        pd.to_pickle(self.kbid_to_predicates, PICKLE_PATH + 'KBID_TO_PREDICATES.pkl')
        pd.to_pickle(self.idx_to_type, PICKLE_PATH + 'IDX_TO_TYPE.pkl')
        pd.to_pickle(self.type_to_idx, PICKLE_PATH + 'TYPE_TO_IDX.pkl')
        logger.info('Process Pickle File Finish.')
