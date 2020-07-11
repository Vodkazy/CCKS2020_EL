#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:53
  @ Author   : Vodka
  @ File     : EntityLinkingPredictor .py
  @ Software : PyCharm
"""
from EntityLinkingModel import EntityLinkingModel
from config import *
from EntityLinkingProcessor import EntityLinkingProcessor

class EntityLinkingPredictor:

    def __init__(self, ckpt_name, batch_size=8, use_pickle=True):
        self.ckpt_name = ckpt_name
        self.batch_size = batch_size
        self.use_pickle = use_pickle

    def generate_tsv_result(self, tsv_name, tsv_type='Valid'):
        processor = EntityLinkingProcessor()
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        if tsv_type == 'Valid':
            examples = processor.get_dev_examples(TSV_PATH + tsv_name)
        elif tsv_type == 'Test':
            examples = processor.get_test_examples(TSV_PATH + tsv_name)
        else:
            raise ValueError('tsv_type error')
        dataloader = processor.create_dataloader(
            examples=examples,
            tokenizer=tokenizer,
            max_length=384,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )

        model = EntityLinkingModel.load_from_checkpoint(
            checkpoint_path=CKPT_PATH + self.ckpt_name,
        )
        model.to(DEVICE)
        model = nn.DataParallel(model)
        model.eval()

        result_list, logit_list = [], []
        for batch in tqdm(dataloader):
            for i in range(len(batch)):
                batch[i] = batch[i].to(DEVICE)

            input_ids, attention_mask, token_type_ids, labels = batch
            logits = model(input_ids, attention_mask, token_type_ids)
            preds = (logits > 0).int()

            result_list.extend(preds.tolist())
            logit_list.extend(logits.tolist())

        tsv_data = pd.read_csv(TSV_PATH + tsv_name, sep='\t')
        tsv_data['logits'] = logit_list
        tsv_data['result'] = result_list
        result_name = tsv_name.split('.')[0] + '_RESULT.tsv'
        tsv_data.to_csv(RESULT_PATH + result_name, index=False, sep='\t')
