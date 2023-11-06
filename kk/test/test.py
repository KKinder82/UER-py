import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import copy
import argparse
from argparse import Namespace
#from uer.layers.cross_layers import CrossVector
#from kk.utils import *
import sentencepiece as spm
# from uer.utils.tokenizers import *
import random
# from finetune.run_c3 import MultipleChoice

def kk_gen():
    try:
        print("第一次运行")
        x = yield 1
        print("1.收到 x={}".format(x))
        print("第二次运行")
        x = yield 2
        print("2.收到 x={}".format(x))
    except Exception as e:
        print("已经触发了 异常。")
        x = yield 9
        print("9.收到 x={}".format(x))
    print("结束")
    x = yield 10
    print("10.收到 x={}".format(x))


def main():
    o = torch.tensor([[1, 2, 3], [5, 6, 7]], dtype=torch.float32)
    y = torch.tensor([[0, 0, 1], [5, 6, 7]], dtype=torch.float32)

    print(" >>> MSELoss <<< ")  # 1/n * sum((o-y)^2)
    loss_fn = nn.MSELoss()
    loss = loss_fn(o, y)
    print(loss)
    print((o-y).pow(2).mean())
    print("-" * 100)

    o = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=torch.float32)
    y = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float32)
    print(" >>> BCELoss <<< ")
    loss_fn = nn.BCELoss()
    loss = loss_fn(o, y)
    print(loss)
    loss_user = -(y * torch.log(o) + (1 - y) * torch.log(1 - o)).mean()
    print(loss_user)
    print("-" * 100)

    o = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=torch.float32)
    y = torch.tensor([[0, 0.1, 0.9], [0.1, 0.8, 0.1]], dtype=torch.float32)
    print(" >>> CrossEntropyLoss <<< ")
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, y)
    print(loss)
    o = F.softmax(o, 1)
    loss_user = -(y * torch.log(o)).sum(1).mean()
    print(loss_user)
    print("-" * 100)

    o = torch.tensor([[2, 0, 0], [0, 0, 4], [0, 1, 0]], dtype=torch.float32)
    y = torch.tensor([0, 2, 1], dtype=torch.long)
    print(" >>> NLLLoss <<< ")
    loss_fn = nn.NLLLoss()
    loss = loss_fn(o, y)
    print(loss)
    loss_user = -(o[y]).sum(1).mean()
    print(loss_user)
    print("-" * 100)

    exit()

    try:
        gen = kk_gen()
        x = gen.send(None)
        print("out1.收到 x={}".format(x))
        x = gen.throw(Exception("我是异常"))
        print("out2.收到 x={}".format(x))
        # x = gen.send("cc")
        # print("3.收到 x={}".format(x))

    except Exception as e:
        print("out4.已经触发了 {}".format(type(e)))
    exit()



    # 组标准化 对一个样本的几个特征（参数指定）的所有数据进行标准化 shape(n,c,d) -> 计算 n*c/个数 个均值与方差
    x = torch.arange(1 * 2 * 3, dtype=torch.float32).view(1, 2, 3) + 1
    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 2, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 3, 2)
    norm = nn.GroupNorm(1 ,3)   # 指定组数、通道数
    y = norm(x)
    # print(x)
    print(y)
    print('-'*100)
    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 3, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 3, 2)
    norm = nn.GroupNorm(1 ,3)   # 指定组数、通道数
    y = norm(x)
    # print(x)
    print(y)

    exit()

    # 实例准化(同层标准化）  一个样本的一个特征的所有数据进行标准化:  shape(n,c,d） -> 计算 n*c 个均值与方差
    x = torch.arange(1 * 2 * 3, dtype=torch.float32).view(1, 2, 3) + 1
    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 2, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 2, 3)
    norm = nn.InstanceNorm1d(3)
    y = norm(x)
    # print(x)
    print(y)
    print('-'*100)
    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 3, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 2, 3)
    norm = nn.InstanceNorm1d(1)
    y = norm(x)
    # print(x)
    print(y)
    print('-'*100)
    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 3, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 2, 3)
    norm = nn.LayerNorm(3)
    y = norm(x)
    # print(x)
    print(y)

    exit()

    # 层标准化  一个样本的一个特征的所有数据进行标准化:  shape(n,c,d） -> 计算 n*c 个均值与方差
    x = torch.arange(1 * 2 * 3, dtype=torch.float32).view(1, 2, 3) + 1
    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 2, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 2, 3)
    norm = nn.LayerNorm(3)
    y = norm(x)
    # print(x)
    print(y)

    print('-'*100)

    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 3, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 2, 3)
    norm = nn.LayerNorm(3)
    y = norm(x)
    # print(x)
    print(y)

    exit()

    # 批归一  （一个特征的所有数据（一个批次下（所有样本）的所有数据）进行标准化:  shape(n,c,d） -> 计算 c个均值与方差）
    x = torch.arange(1 * 2 * 3, dtype=torch.float32).view(1, 2, 3) + 1
    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 2, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 2, 3)
    norm = nn.BatchNorm1d(2)
    y = norm(x)
    # print(x)
    print(y)

    print('-'*100)

    x = torch.tensor([[1, 2, 3, 2, 3, 5, ], [2, 2, 5, 2, 2, 2, ]], dtype=torch.float32)
    x = x.view(2, 2, 3)
    norm = nn.BatchNorm1d(2)
    y = norm(x)
    # print(x)
    print(y)

    exit(0)


    x = [i for i in range(1,5)]
    print(x)
    exit()



    x = torch.tensor([])
    
    # 创建一个示例张量
    x = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    # 选择第一维的所有元素
    y = x[slice(None), 1]
    print(y)  # 输出: tensor([2, 5, 8])
    y = x[:, 1]
    print(y)  # 输出: tensor([2, 5, 8])
    y = x[slice(None), 0]
    print(y)  # 输出: tensor([1, 4, 7])

    # 选择第二维的所有元素
    z = x[0, slice(None)]
    print(z)  # 输出: tensor([1, 2, 3])
    z = x[1, slice(None)]
    print(z)  # 输出: tensor([4, 5, 6])

    print([slice(None)] * 3)
    exit(0)

    # 在设备0上创建张量x
    x0 = torch.tensor([[1, 2, 3],
                       [4, 5, 6]], device='cuda:0')

    # 在设备1上创建张量x
    x1 = torch.tensor([[1, 8, 9],
                       [10, 15, 12]], device='cuda:0')

    # 创建目标设备（设备0）上的张量
    target_device = torch.device('cuda:0')
    target_shape = (2, 3)
    target_tensor = torch.zeros(target_shape, device=target_device)
    target_tensor = torch.arange(1, 13, device=target_device).reshape(4, 3).float()
    print(target_tensor)

    indexes = torch.tensor([[0, 1], [0, 2]], device=target_device)
    out = torch.gather(target_tensor, 0, indexes)
    print(out)

    exit(0)

    # 创建目标设备（设备0）上的张量



    x = torch.arange(3, 23).float().reshape(-1, 5)
    print(x)
    x[:, 2] = 100
    x1 = nn.Softmax(dim=-1)(x)
    print(x1)
    x1 = torch.log(x1)
    print(x1)
    x = nn.LogSoftmax(dim=-1)(x)
    print(x)
    y = torch.tensor([1, 2, 1, 1])


    loss = nn.NLLLoss(reduction="mean")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(loss)
    loss = loss(x, y)
    print(loss)

    loss = nn.NLLLoss(reduction="none")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(loss)
    loss = loss(x, y)
    print(loss)

    exit()

    # a  = torch.arange(80).reshape(2,2,2,10).float()
    # cv = CrossVector(10, 1)
    # print(cv(a))
    # exit()
    #
    # args = {"spm_model_path": r"E:\Data\AiModel\chatglm-6b\ice_text.model", "vocab_path": "uer/google_zh_vocab.txt", "token_len":50}
    # args = {"spm_model_path": r"", "vocab_path": "uer/google_zh_vocab.txt", "do_lower_case":True, "token_len":50}
    # args = {"spm_model_path": r"", "vocab_path": "uer/chatGLM6_vocab.txt", "do_lower_case":True, "token_len":50}
    # args = {"spm_model_path": r"", "vocab_path": "uer/kk_zh_vocab.txt", "do_lower_case":True, "token_len":50}
    # args = Namespace(**args)
    # # token = KKTokenizer(args)
    # token = KKTokenizer(args)
    # input = "中国人民解放军是一支战无不胜的队伍1335"
    # out = token.tokenize(input) + [SEP_TOKEN]
    # print(out)
    # out = token.convert_tokens_to_ids(out)
    # print(out)
    # out = token.convert_ids_to_tokens(out)
    # print(out)
    # print("------------")
    #
    # args = {"spm_model_path": r"E:\Data\AiModel\chatglm-6b\ice_text.model", "vocab_path": "", "do_lower_case":True, "token_len":50}
    # args = Namespace(**args)
    # token = KKTokenizer(args)
    # input = "中国人民解放军是一支战无不胜的队伍"
    # out = token.tokenize(input) + [SEP_TOKEN]
    # print(out)
    # out = token.convert_tokens_to_ids(out)
    # print(out)
    # out = token.convert_ids_to_tokens(out)
    # print(out)
    #
    # exit()

    # a = {"a":"AAA", "b":"BBBB"}
    # if "a" in a:
    #     print("OK")
    # if "c" in a:
    #     print("F")
    # exit()
    #
    # sp_model = spm.SentencePieceProcessor(args)
    #
    # sp_model.Load(r"E:\Data\AiModel\chatglm-6b\ice_text.model")
    # a = sp_model.EncodeAsPieces("<pad>中国人民解放军是一支战无不胜的队伍")
    # pad_str = sp_model.IdToPiece(sp_model.pad_id())
    # print(pad_str)

    # with open("d:/icon_text_vocab.txt", "w", encoding="utf-8") as f:
    #     for i in vocab:
    #         f.write(i + "\n")
    exit()


    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _tokenize_chinese_chars(text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if _is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    input = "中国人和USA不一样"
    out = _tokenize_chinese_chars(input)
    print(input)
    exit()




    #
    # args = load_argsconfig("test.txt")
    # print(args)
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_mode_path", default="test.txt", type=str, help="The path of pretrained model.")
    # args = parser.parse_args(args)
    # print(args)
    #
    # exit()

    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # args = parser.parse_args()
    # model = MultipleChoice(args)
    model = nn.Linear(10, 10)
    ps = model.named_parameters()
    for i in ps:
        print(i)
    exit()

    # a = [1, 2 , 3]
    # b = [4, 5, 6]
    # print(a + b)
    # exit(0)
    #
    # feather_size = 5
    # x = torch.arange(3,2 * feather_size + 3).float().reshape(-1, feather_size)
    # cross_vecter = CrossVector(feather_size)
    # x_out = cross_vecter(x)
    # print(x_out)
    # exit()

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
    l = [1, 2, 3]
    print(l * 3)
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


    x = torch.tensor([1, 2, 3]).float()
    y = torch.tensor([1230]).float()

    model = testLayer()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_function = torch.nn.MSELoss()
    print("---- 1 ----")
    print_model(model)
    out = model(x)
    loss = loss_function(out, y)

    print("---- 2 ----")
    print_model(model)
    loss.backward()

    print("---- 3 ----")
    print_model(model)
    optim.step()

    print("---- 4 ----")
    print_model(model)
    optim.zero_grad()

    print("---- 5 ----")
    print_model(model)

    exit(0)

    x = torch.arange(10).view(5, 2).float() * 0.5
    mn = torch.ones(2).float()
    mx = torch.ones(2).float()
    mx[1] = mx[1] + 2
    print(x)

    for b in range(x.shape[0]):
        x[b][x[b] < mn] = mn[x[b] < mn]
        x[b][x[b] > mx] = mx[x[b] > mx]

    print("")
    print(x)

    lr = LambdaLR([])
    for i in range(10):
        x = fun1()
        print(x)
        t += 1

if __name__ == "__main__":
    main()
