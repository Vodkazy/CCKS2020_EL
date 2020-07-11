#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:48
  @ Author   : Vodka
  @ File     : EntityTypingProcessor .py
  @ Software : PyCharm
"""
from config import *


class EntityTypingProcessor(DataProcessor):
    """实体链接数据处理"""

    def get_train_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='train',
        )

    def get_dev_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='valid',
        )

    def get_test_examples(self, file_path):
        return self._create_examples(
            self._read_tsv(file_path),
            set_type='test',
        )

    def get_labels(self):
        return PICKLE_DATA['IDX_TO_TYPE']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            text_a = line[1]
            text_b = line[3]
            label = line[-1]
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label,
            ))
        return examples

    def create_dataloader(self, examples, tokenizer, max_length=64,
                          shuffle=False, batch_size=16, use_pickle=False):
        pickle_name = 'ET_FEATURE_' + examples[0].guid.split('-')[0].upper() + '.pkl'
        if use_pickle:
            features = pd.read_pickle(PICKLE_PATH + pickle_name)
        else:
            features = glue_convert_examples_to_features(
                examples,
                tokenizer,
                label_list=self.get_labels(),
                max_length=max_length,
                output_mode='classification',
            )
            pd.to_pickle(features, PICKLE_PATH + pickle_name)

        dataset = torch.utils.data.TensorDataset(
            torch.LongTensor([f.input_ids for f in features]),
            torch.LongTensor([f.attention_mask for f in features]),
            torch.LongTensor([f.token_type_ids for f in features]),
            torch.LongTensor([f.label for f in features]),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=6,
        )
        return dataloader

    def generate_feature_pickle(self, max_length):
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        train_examples = self.get_train_examples(TSV_PATH + 'ET_TRAIN.tsv')
        valid_examples = self.get_dev_examples(TSV_PATH + 'ET_VALID.tsv')
        test_examples = self.get_test_examples(TSV_PATH + 'ET_TEST.tsv')

        self.create_dataloader(
            examples=train_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=True,
            batch_size=16,
            use_pickle=False,
        )
        self.create_dataloader(
            examples=valid_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=False,
            batch_size=16,
            use_pickle=False,
        )
        self.create_dataloader(
            examples=test_examples,
            tokenizer=tokenizer,
            max_length=max_length,
            shuffle=False,
            batch_size=16,
            use_pickle=False,
        )
