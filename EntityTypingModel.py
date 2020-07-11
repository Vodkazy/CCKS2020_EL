#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/7 上午10:54
  @ Author   : Vodka
  @ File     : EntityTypingModel .py
  @ Software : PyCharm
"""

from EntityTypingProcessor import EntityTypingProcessor
from config import *


class EntityTypingModel(pl.LightningModule):
    """实体类型推断模型"""

    def __init__(self, max_length=64, batch_size=16, use_pickle=True):
        super(EntityTypingModel, self).__init__()
        # 输入最大长度
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_pickle = use_pickle
        # 二分类损失函数
        self.criterion = nn.CrossEntropyLoss()

        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # 预训练模型
        self.bert = BertForSequenceClassification.from_pretrained(
            "hfl/chinese-roberta-wwm-ext",
            num_labels=len(PICKLE_DATA['IDX_TO_TYPE']),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]

    def prepare_data(self):
        self.processor = EntityTypingProcessor()
        self.train_examples = self.processor.get_train_examples(TSV_PATH + 'ET_TRAIN.tsv')
        self.valid_examples = self.processor.get_dev_examples(TSV_PATH + 'ET_VALID.tsv')
        self.test_examples = self.processor.get_test_examples(TSV_PATH + 'ET_TEST.tsv')

        self.train_loader = self.processor.create_dataloader(
            examples=self.train_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=True,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )
        self.valid_loader = self.processor.create_dataloader(
            examples=self.valid_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )
        self.test_loader = self.processor.create_dataloader(
            examples=self.test_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            shuffle=False,
            batch_size=self.batch_size,
            use_pickle=self.use_pickle,
        )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean()

        tensorboard_logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(outputs, labels)

        _, preds = torch.max(outputs, dim=1)
        acc = (preds == labels).float().mean()

        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': val_loss, 'val_acc': val_acc}
        return {'val_loss': val_loss, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-5, eps=1e-8)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
