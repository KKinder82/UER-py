"""
This script provides an exmaple to wrap UER-py for C3 (a multiple choice dataset).
"""
import sys
import os
import argparse
import json
import random
import torch
import torch.nn as nn
import json

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts, adv_opts
from finetune.run_classifier import build_optimizer, load_or_initialize_parameters, train_model, batch_loader, evaluate

from kk.kk_utils import *
from uer.layers.cross_layers import CrossVector


# 多选题
class MultipleChoice(nn.Module):
    def __init__(self, args):
        super(MultipleChoice, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.dropout = nn.Dropout(args.dropout)
        # self.output_layer = nn.Linear(args.hidden_size, 1)
        self.output_layer = CrossVector(args.hidden_size, 1)
        self.nll_loss = nn.NLLLoss()
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x choices_num x seq_length] | 选项 + 问题 + 背景
            tgt: [batch_size] | 0, 1, 2, 3, 答案 序号
            seg: [batch_size x choices_num x seq_length]  # 掩码
        """

        choices_num = src.shape[1]          # 最大候选项数量

        src = src.view(-1, src.size(-1))    # shape(batch_size * choices_num, seq_length)
        seg = seg.view(-1, seg.size(-1))    # shape(batch_size * choices_num, seq_length)

        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        output = self.dropout(output)
        logits = self.output_layer(output[:, 0, :])
        reshaped_logits = logits.view(-1, choices_num) # [batch_size x choices_num]

        if tgt is not None:
            tgt = tgt.view(-1)  # [batch_size]
            _reshaped_logits = self.logSoftmax(reshaped_logits)
            loss = self.nll_loss(_reshaped_logits, tgt)
            return loss, reshaped_logits
        else:
            return None, reshaped_logits

def read_dataset(args, path):

    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for i in range(len(data)):
        # i: 一道题
        for j in range(len(data[i][1])):
            # j: 试题信息
            example = ["\n".join(data[i][0]).lower(), data[i][1][j]["question"].lower()]
            for k in range(len(data[i][1][j]["choice"])):
                # k 一个候选项
                example += [data[i][1][j]["choice"][k].lower()]
            for k in range(len(data[i][1][j]["choice"]), args.max_choices_num):
                example += ["No Answer"]

            example += [data[i][1][j].get("answer", "").lower()]

            examples += [example]
    # example[[0背景，1问题，2选项1， 3选项2， 4选项3， 5选项4，6答案], []]
    dataset = []
    # dataset = [([src|候选项1,], tgt|答案序号, [seg|掩码,]) ]  一个候选项
    for i, example in enumerate(examples):
        tgt = 0
        # 将答案输换为 0,1,2,3 的序号
        for k in range(args.max_choices_num):
            if example[2 + k] == example[6]:
                tgt = k     # 序号 0,1,2,3
        dataset.append(([], tgt, []))

        for k in range(args.max_choices_num):
            # src_a -> CLS_TOKEN + 选项 + SEP_TOKEN
            src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(example[k + 2]) + [SEP_TOKEN])
            # src_b -> 问题 + SEP_TOKEN
            src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(example[1]) + [SEP_TOKEN])
            # src_b -> 背景 + SEP_TOKEN
            src_c = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(example[0]) + [SEP_TOKEN])

            # src -> CLS_TOKEN + 选项 + SEP_TOKEN + 问题 + SEP_TOKEN + 背景 + SEP_TOKEN | 背景 相对来说 不重要, 可以删除，因此，后在后面
            src = src_a + src_b + src_c
            # seg 1: 选项 + 问题 | 2: 背景
            seg = [1] * (len(src_a) + len(src_b)) + [2] * len(src_c)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]

            # Fill Padding
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)

            dataset[-1][0].append(src)
            dataset[-1][2].append(seg)

    return dataset


def main():
    config_args = []
    config_args = kk_load_argsconfig("afinetune/run_c3.txt")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--max_choices_num", default=4, type=int,
                        help="The maximum number of cadicate answer, shorter than this will be padded.")

    tokenizer_opts(parser)

    adv_opts(parser)

    # 分析参数
    args = parser.parse_args(args=config_args)

    args.labels_num = args.max_choices_num

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build multiple choice model.
    model = MultipleChoice(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    # Get logger.
    args.logger = init_logger(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    # Training phase.
    # trainset [topic_infos|试题信息, answer_index|答案,  segs|分段]
    #   topic_infos: (batch * 最大选项个数, sentence_length) | [选项 + 问题 + 背景]
    #   answer_index: (batch ,) | 答案序号 | 0, 1, 2, 3
    #   segs: (batch * 最大选项个数, sentence_length) | 1,1,1, 2,2,2

    trainset = read_dataset(args, args.train_path)
    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    if args.use_adv:
        args.adv_method = str2adv[args.adv_type](model)

    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")

    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.LongTensor([example[0] for example in trainset])
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg = torch.LongTensor([example[2] for example in trainset])

        model.train()
        for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):

            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch)
            # 从Tensor对象 取出 值
            total_loss += loss.item()

            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    # Evaluation phase.
    if args.test_path is not None:
        args.logger.info("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            args.model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            args.model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    main()
