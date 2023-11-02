from uer.embeddings import *
from uer.encoders import *
from uer.decoders import *
from uer.targets import *
from uer.models.model import Model


def build_model(args):
    """
    Build universial encoder representations uer.
    The combinations of different embedding, encoder,
    and target layers yield pretrained uer of different
    properties.
    We could select suitable one for downstream tasks.
    """

    # Token 嵌入对象
    embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))

    # 创建编码器
    encoder = str2encoder[args.encoder](args)

    # 创建目标对象解码器
    if args.decoder is not None:
        if args.data_processor == "mt":
            tgt_embedding = str2embedding[args.tgt_embedding](args, len(args.tgt_tokenizer.vocab))
        else:
            tgt_embedding = str2embedding[args.tgt_embedding](args, len(args.tokenizer.vocab))
        decoder = str2decoder[args.decoder](args)
    else:
        tgt_embedding = None
        decoder = None

    # 创建目标对象处理器
    target = Target()
    for target_name in args.target:
        if args.data_processor == "mt":
            tmp_target = str2target[target_name](args, len(args.tgt_tokenizer.vocab))
        else:
            tmp_target = str2target[target_name](args, len(args.tokenizer.vocab))
        target.update(tmp_target, target_name)

    #
    model = Model(args, embedding, encoder, tgt_embedding, decoder, target)

    return model
