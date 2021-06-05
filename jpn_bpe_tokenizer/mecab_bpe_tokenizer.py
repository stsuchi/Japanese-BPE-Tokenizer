from transformers import RobertaTokenizer
from tokenizers import AddedToken
from typing import Optional
import os
import unicodedata

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

get_pairs('lowest')
list('lowest')

class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""

    def __init__(
        self,
        do_lower_case=False,
        never_split=None,
        normalize_text=True,
        mecab_dic: Optional[str] = "ipadic",
        mecab_option: Optional[str] = None,
    ):
        """
        Constructs a MecabTokenizer.
        Args:
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of tokens not to split.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **mecab_dic**: (`optional`) string (default "ipadic")
                Name of dictionary to be used for MeCab initialization. If you are using a system-installed dictionary,
                set this option to `None` and modify `mecab_option`.
            **mecab_option**: (`optional`) string
                String passed to MeCab constructor.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text

        try:
            import fugashi
        except ModuleNotFoundError as error:
            raise error.__class__(
                "You need to install fugashi to use MecabTokenizer. "
                "See https://pypi.org/project/fugashi/ for installation."
            )

        mecab_option = mecab_option or ""

        if mecab_dic is not None:
            if mecab_dic == "ipadic":
                try:
                    import ipadic
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The ipadic dictionary is not installed. "
                        "See https://github.com/polm/ipadic-py for installation."
                    )

                dic_dir = ipadic.DICDIR

            elif mecab_dic == "unidic_lite":
                try:
                    import unidic_lite
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The unidic_lite dictionary is not installed. "
                        "See https://github.com/polm/unidic-lite for installation."
                    )

                dic_dir = unidic_lite.DICDIR

            elif mecab_dic == "unidic":
                try:
                    import unidic
                except ModuleNotFoundError as error:
                    raise error.__class__(
                        "The unidic dictionary is not installed. "
                        "See https://github.com/polm/unidic-py for installation."
                    )

                dic_dir = unidic.DICDIR
                if not os.path.isdir(dic_dir):
                    raise RuntimeError(
                        "The unidic dictionary itself is not found."
                        "See https://github.com/polm/unidic-py for installation."
                    )

            else:
                raise ValueError("Invalid mecab_dic is specified.")

            mecabrc = os.path.join(dic_dir, "mecabrc")
            mecab_option = f'-d "{dic_dir}" -r "{mecabrc}" ' + mecab_option

        self.mecab = fugashi.GenericTagger(mecab_option)

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize("NFKC", text)

        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []

        for word in self.mecab(text):
            token = word.surface

            if self.do_lower_case and token not in never_split:
                token = token.lower()

            tokens.append(token)

        return tokens


class MecabBPETokenizer(RobertaTokenizer):
    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=False,
        **kwargs
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super(RobertaTokenizer,self).__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self.word_tokenizer = MecabTokenizer(mecab_dic='unidic_lite')

    def _tokenize(self,text):
        tokens = self.word_tokenizer.tokenize(text, never_split=self.all_special_tokens)

        bpe_tokens=[self.bpe(token).split() for token in tokens]
        bpe_tokens=[i for j in bpe_tokens for i in j]

        return bpe_tokens
