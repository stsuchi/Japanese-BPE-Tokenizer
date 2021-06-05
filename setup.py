from setuptools import setup

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='Japanese-BPE-Tokenizer',
      version='0.1.1',
      description='Mecab-based BPE toknizer for Japanese text',
      url='http://github.com/stsuchi/Japanese-BPE-Tokenizer',
      author='Shiro T.',
      license='MIT',
      packages=['jpn_bpe_tokenizer'],
      install_requires=['transformers','fugashi','unidic_lite'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      zip_safe=False)
