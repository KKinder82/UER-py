import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import copy
import argparse
from argparse import Namespace
from uer.layers.CrossLayers import CrossVector
from kk.utils import *
import sentencepiece as spm
from uer.utils.tokenizers import *
import random
# from finetune.run_c3 import MultipleChoice


def main():
    args = {"spm_model_path": r"E:\Data\AiModel\chatglm-6b\ice_text.model", "vocab_path": ""}
    args = Namespace(**args)
    token = KKTokenizer(args)
    input = "中国人民解放军是一支战无不胜的队伍[MASK]"
    out = token.tokenize(input)
    print(out)
    out = token.convert_tokens_to_ids(out)
    print(out)
    exit()

    sp_model = spm.SentencePieceProcessor(args)

    sp_model.Load(r"E:\Data\AiModel\chatglm-6b\ice_text.model")
    a = sp_model.EncodeAsPieces("<pad>中国人民解放军是一支战无不胜的队伍")
    pad_str = sp_model.IdToPiece(sp_model.pad_id())
    print(pad_str)

    # with open("d:/icon_text_vocab.txt", "w", encoding="utf-8") as f:
    #     for i in vocab:
    #         f.write(i + "\n")
    exit()

    x = torch.arange(3, 18).float().reshape(-1, 5)
    print(x)
    x[:, 2] = 100
    x1 = nn.Softmax(dim=-1)(x)
    print(x1)
    x1 = torch.log(x1)
    print(x1)
    x = nn.LogSoftmax(dim=-1)(x)
    print(x)
    y = torch.tensor([1,2,1])
    loss = nn.NLLLoss()(x, y)
    print(loss)
    exit()

    args = load_argsconfig("test.txt")
    print(args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_mode_path", default="test.txt", type=str, help="The path of pretrained model.")
    args = parser.parse_args(args)
    print(args)

    exit()

    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # args = parser.parse_args()
    # model = MultipleChoice(args)
    model = nn.Linear(10, 10)
    ps = model.named_parameters()
    for i in ps:
        print(i)
    exit()


    a = [1, 2 , 3]
    b = [4, 5, 6]
    print(a + b)
    exit(0)

    feather_size = 5
    x = torch.arange(3,2 * feather_size + 3).float().reshape(-1, feather_size)
    cross_vecter = CrossVector(feather_size)
    x_out = cross_vecter(x)
    print(x_out)
    exit()

    _one = x.new_ones((1, 1)).expand(x.size(0), 1)
    x_1 = x[..., None]
    print(x_1)
    x_t = x_1.transpose(-1, -2)
    print(x_t)
    x_out = torch.bmm(x_1, x_t).tril()
    print(x_out)
    x_out = x_out.flatten(start_dim=-2)
    print(x_out)
    x_out = torch.cat((_one, x, x_out), dim=-1)
    print(x_out)
    exit()

    x = torch.arange(3,8).float().reshape(-1, 5)
    x = x.unsqueeze(1)
    print(x)
    x_t = x.transpose(-1, -2)
    print(x_t)

    _one = x.new_ones((1, 1, 1)).expand(x.size(0), 1, 1)
    print(_one)

    x = torch.cat([_one, x], dim=-1)
    print(x)

    x_out = torch.bmm(x_t, x )
    print(x_out)
    exit()


    class aclass():
        def __len__(self):
            return 1

        def __iter__(self):
            yield self.Status
            raise StopIteration

        def __init__(self, status:int):
            self.Status = status

    class tclass():
        def __init__(self, name, age):
            self.Name = name
            self.age = age
            self.stream_0 = aclass(age)

    args = tclass("zhangsan", 18)
    print(args.__dict__)
    exit()

    stream_0_args = copy.deepcopy(vars(args))
    # stream_0_args.update(args.stream_0)
    stream_1_args = Namespace(**stream_0_args)
    print(stream_0_args)
    exit(0)

    from  uer.layers.relative_position_embedding import RelativePositionEmbedding
    encode = torch.arange(20).float().reshape(4, 5)
    decode = torch.arange(20).float().reshape(4, 5)
    layer = RelativePositionEmbedding(2)
    x = layer.forward(encode, decode)
    print(x)
    exit(0)


    a = torch.arange(5).reshape(5, 1)
    b = torch.arange(5).reshape(1, 5)
    print( a - b)
    exit(0)

    a = torch.arange(1, 13).float().reshape(3, 4)
    print(a)
    print(a * 3)
    list = [1, 2, 3]
    print(list * 3)
    exit(0)

    a : torch.tensor = torch.arange(1, 13).float().reshape(3, 4)
    print(a.is_contiguous())
    print("a",a.is_contiguous(), id(a), id(a.untyped_storage()))

    print( "-"*60)
    b = a[1:, 0:-1:2]
    print("b",b.is_contiguous(), id(b), id(b.untyped_storage()))

    b.contiguous()
    print("b", b.is_contiguous(), id(b), id(b.untyped_storage()))

    b = b.contiguous()
    print("b", b.is_contiguous(), id(b), id(b.untyped_storage()))

    print("-" * 60)
    c = b.contiguous().T
    print("c", c.is_contiguous(), id(c), id(c.untyped_storage()))
    exit(0)

    # self.encoder_0 = str2encoder[stream_0_args.encoder](stream_0_args)


    hidden_size = 5
    conv_b1 = nn.Parameter(torch.randn(1, hidden_size, 1, 1))
    print(conv_b1)
    exit()

    bs = (
            list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    ds = [n for n in cs]
    cs = [chr(n) for n in cs]
    t = dict(zip( cs, ds))
    print(t)
    exit(0)

    # 加载 SentencePiece 模型
    spm_model_path = "path/to/your/spm_model.model"
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(spm_model_path)

    # 要编码的文本
    text = "I love natural language processing"

    # 使用 SentencePiece 对文本进行编码
    encoded_pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)

    # 打印编码结果
    print(encoded_pieces)

    exit(0)

    inputs = "this is a goog student ? "
    outputs = inputs.strip().split()
    print(outputs)
    outputs = " ".join(outputs)

    print(outputs)
    exit()

    t = 1
    fun1 = lambda t : t * 2

    lr = LambdaLR([])
    for i in range(10):
        x = fun1()
        print(x)
        t += 1

if __name__ == "__main__":
    main()
