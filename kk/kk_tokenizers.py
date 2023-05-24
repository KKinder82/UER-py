import torch
import torch.nn as nn
from uer.utils.tokenizers import Tokenizer


class KKTokenizer(Tokenizer):
    """
    """
    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)
        if not args.spm_model_path:
            raise ValueError("Please specify a vocabulary file path by --spm_model_path.")

    def tokenize(self, text):
        return self.sp_model.SampleEncodeAsPieces(text, 64)

