# Byte Pair Encoding for Japanese Language

## Summary
The package applies Mecab and Byte Pair Encoding algorithms to tokenize Japanese text.

## Usage
To train a new tokenizer, import the module
```
from train.implementations import MecabBPETrainTokenizer
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

And finally, save the trained config files
```
tokenizer.save_model('config')
```
