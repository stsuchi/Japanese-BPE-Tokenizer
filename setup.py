from setuptools import setup

setup(name='Japanese-BPE-Tokenizer',
      version='0.1',
      description='Mecab-based BPE toknizer for Japanese text',
      url='http://github.com/stsuchi/Japanese-BPE-Tokenizer',
      author='Shiro T.',
      license='MIT',
      packages=['jpn_bpe_tokenizer'],
      install_requires=['transformers','fugashi','unidic_lite'],
      zip_safe=False)
