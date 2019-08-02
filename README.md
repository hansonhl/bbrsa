# Batch and Beam RSA (BBRSA)

Implementation of the Rational Speech Acts model, using efficient beam search and batched methods, inspired by the OpenNMT framework.

## Prerequisites
__Dependencies__

```
pytorch 1.0
numpy
opennmt
pytorch-transformers
```

### Installing OpenNMT

To install OpenNMT, follow the following steps:

1. Clone the OpenNMT pytorch repository
```
git clone https://github.com/OpenNMT/OpenNMT-py
```
2. Install the package in pip development mode:
```
cd OpenNMT-py
pip install -e .
```

In this way `pip` will automatically detect the python package `onmt`, and you
will be able to `import onmt` and its modules in any python script.

### Installing the `pytorch-transformers` API

`pytorch-transformers` refers to the Huggingface pytorch implementation of BERT. It is only required for distractor generation. 

When installing, please manually clone the directory and install locally. As for now, __DO NOT__ install through pip as it is outdated, install it in a similar way to how OpenNMT was installed as shown above. See [this link](https://github.com/huggingface/pytorch-transformers) for more details.
