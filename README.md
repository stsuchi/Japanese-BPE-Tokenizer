# Byte Pair Encoding for Japanese Language

## Summary
The package applies Mecab and Byte Pair Encoding algorithms to tokenize Japanese text.

## Usage
To train a new tokenizer, import the module
```
from jpn_bpe_tokenizer import MecabBPETrainTokenizer
```

Instantiate the object
```
tokenizer = MecabBPETrainTokenizer()
```

Set the arguments with the text, vocab_size and special_tokens to train the tokenizer
```
tokenizer.train(files,
                vocab_size=52000,
                special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"])
```

Save the trained config files
```
tokenizer.save_model('config')
```

When you have config files, use them to tokenize new text with a tokenizer object
```
from jpn_bpe_tokenizer import MecabBPETokenizer

tokenizer = MecabBPETokenizer('config/vocab.json','config/merges.txt')
```
