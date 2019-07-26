# Batch and Beam RSA (BBRSA)

Implementation of the Rational Speech Acts model, using efficient beam search and batched methods, inspired by the OpenNMT framework.

## Prerequisites
__Dependencies__

```
pytorch
numpy
opennmt
pytorch-transformers
```

`pytorch-transformers` refers to the Huggingface pytorch implementation of BERT. It is required for distractor generation. When installing, please manually clone the directory and install locally. As for now, __DO NOT__ install through pip as it is outdated. See [this link](https://github.com/huggingface/pytorch-transformers) for more details.

OpenNMT cannot be installed directly through pip. Simply clone the OpenNMT pytorch repository ```git clone https://github.com/OpenNMT/OpenNMT-py``` Then in `bbrsa/__init__.py` set the variable `ONMT_DIR = <path to local OpenNMT repo>`.
