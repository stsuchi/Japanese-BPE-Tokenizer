from tokenizers import AddedToken, normalizers, pre_tokenizers
from tokenizers.implementations import ByteLevelBPETokenizer
from .pre_tokenizers import MeCabPreTokenizer
import os

class MecabBPETrainTokenizer(ByteLevelBPETokenizer):
    def __init__(self,vocab=None,merges=None):
        super().__init__(vocab=vocab,merges=merges)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self._tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Strip()])
        # if use the custom normalizer
        #self._tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Strip(),normalizers.Normalizer.custom(CustomNormalizer())])

        self._tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(MeCabPreTokenizer('unidic_lite'))
