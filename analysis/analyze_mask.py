import torch, logging, argparse, pickle, time
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from collections import defaultdict as DD

from more_itertools import chunked

def main():
    logging.basicConfig(level=logging.WARNING)
    def_top_k = 10
    parser = argparse.ArgumentParser(
        description='''Script to find out what words BERT\'s
        attention attends to in layers 0, 9, 10, 11.''')
    parser.add_argument('-s', '--src', type=argparse.FileType('r'), metavar='PATH',
        help='''File containing multiple lines of input text, if not specified,
              uses some predefined text.''')
    parser.add_argument('-t', '--top', type=int, default=def_top_k, metavar='K',
        help='Find top K words that BERT attends to. Default 10.')
    parser.add_argument('-b', '--batch_size', type=int, metavar='B',
        help='Specify batch size=B. Will process items one by one if not set.')
    parser.add_argument('-g', '--gpu', action='store_true',
        help='Option to use GPU.')
    # parser.add_argument('-o', '--out', type=argparse.FileType('w'), \
    #     default='top_' + str(def_top_k) + 'attended.txt')
    args = parser.parse_args()

    top_k = args.top
    batch_size = args.batch_size
    use_gpu = args.gpu

if __name__ == '__main__':
    main()
