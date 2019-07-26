"""
Discover patterns in what BERT attends to. 'What does Bert look at?' paper shows
that there is a high entropy in beginning and ending layers' attention of BERT.
I would like to know about the distribution of words attended to by BERT at
these layers, if BERT were given news article sentences form the Gigaword
summary dataset.

In this script I look at attentions in layers 0, 9, 10, 11 given a sentence.
I sum over the attentions over all layers and heads and normalize it to get a
distribution, and record what words get the most attention.
"""


import torch, logging, argparse, pickle, time
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from collections import defaultdict as DD

from more_itertools import chunked

def batch_to_idx_tensor(tokenizer, text):
    assert isinstance(text, list), 'Must input a list of strings!'

    str_tokens = [tokenizer.tokenize('[CLS] ' + t.strip() + ' [SEP]') for t in text]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(t) for t in str_tokens]
    seq_lens = torch.LongTensor(list(map(len, indexed_tokens)))
    seq_tensor = torch.zeros((len(indexed_tokens), seq_lens.max()), dtype=torch.long)
    attn_mask_tensor = torch.zeros((len(indexed_tokens), seq_lens.max()), dtype=torch.long)
    pad_idx = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    seq_tensor.fill_(pad_idx)

    for idx, (seq, seqlen) in enumerate(zip(indexed_tokens, seq_lens)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        attn_mask_tensor[idx, :seqlen] = 1

    scrm_seq_lens, scrm_idxs = seq_lens.sort(0, descending=True)
    scrm_seq_tensor = seq_tensor[scrm_idxs]
    scrm_str_tokens = [str_tokens[i.item()] for i in scrm_idxs]
    scrm_attn_mask = attn_mask_tensor[scrm_idxs]

    return scrm_seq_tensor, scrm_str_tokens, scrm_attn_mask, scrm_idxs

def str_to_idx_tensor(tokenizer, text, masked_words=None):
    assert isinstance(text, str), 'Must input a string!'
    # tokens_tensor = torch.tensor(indexed_tokens)
    text = '[CLS] ' + text + ' [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    # print(tokenized_text)

    if masked_words is not None:
        for i, tok in enumerate(tokenized_text):
            if tok in masked_words:
                tokenized_text[i] = '[MASK]'

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    return tokens_tensor

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

    if args.src is not None:
        src = args.src
    else:
        text = 'burma has put five cities on a security alert after religious unrest involving buddhists and moslems in the northern city of mandalay , an informed source said wednesday.'
        text1 = 'police arrested five anti-nuclear protesters friday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .'
        text2 = 'turkmen president gurbanguly berdymukhammedov will begin a two-day visit to russia , his country \'s main energy partner , on monday for trade talks , the kremlin press office said .'
        text3 = 'israel \'s new government barred yasser arafat from flying to the west bank to meet with former prime minister shimon peres on thursday , a move palestinian officials said violated the israel-plo peace accords .'
        src = [text, text1, text2, text3]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()

    if use_gpu:
        assert torch.cuda.is_available(), 'GPU unavailable!'
        model = model.cuda()
        print('-- Using GPU --')
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    attended_word_dict = DD(int)
    total_words_attended = 0

    total_start_time = time.time()

    if batch_size is None:
        count = 0
        layers = [0, 9, 10, 11]
        for line in src:
            if count % 100 == 0:
                print('Processed', count, 'lines')
            count += 1
            line = line.strip()
            tokens_tensor = str_to_idx_tensor(tokenizer, line)
            if use_gpu:
                tokens_tensor = tokens_tensor.cuda()
            str_tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist()[0])
            outputs = model(tokens_tensor)

            cum_attn = []

            for l in layers:
                layer = outputs[2][l]
                summed = layer.sum(dim=2).sum(dim=1).view(-1)
                summed = (summed / summed.sum(dim=0))
                cum_attn.append(summed)

            all_attns = torch.stack(cum_attn).sum(dim=0)
            normalized_attn = (all_attns / all_attns.sum(dim=0)).tolist()
            sorted_by_attn = sorted(list(zip(normalized_attn, str_tokens)), key=lambda p: p[0], reverse=True)

            for p in sorted_by_attn[:top_k]:
                attended_word_dict[p[1]] += 1

            total_words_attended += len(sorted_by_attn[:top_k])
        if args.src is not None:
            args.src.close()
    else:
        # batch implementation
        layers = torch.tensor([0, 9, 10, 11], device=device)
        attended_word_tensor = torch.zeros(tokenizer.vocab_size, device=device)
        batch_iterator = chunked(src, batch_size)
        iter = 0

        for batch in batch_iterator:
            start_time = time.time()
            if iter % 10 == 0:
                print('Processed', iter, 'batches')
            iter += 1

            batch_toks, _, attn_mask, _ = batch_to_idx_tensor(tokenizer, batch)

            if use_gpu:
                batch_toks = batch_toks.cuda()
                attn_mask = attn_mask.cuda()

            with torch.no_grad():
                outputs = model(batch_toks, attention_mask=attn_mask)

            attn = torch.stack(outputs[2]).index_select(0, layers)
            summed = attn.sum(dim=3).sum(dim=2).sum(dim=0).view(attn.shape[1], attn.shape[4])
            summed = summed / summed.sum(dim=1, keepdim=True)

            _, topk_idxs = summed.topk(top_k, sorted=False)

            split_batch_toks = batch_toks.split(1, dim=0)
            split_topk_idxs = topk_idxs.split(1, dim=0)

            for idxs, toks in zip(topk_idxs, split_batch_toks):
                attended_toks = toks.view(-1)[idxs.view(-1)]
                attended_word_tensor[attended_toks] += 1

            total_words_attended += topk_idxs.shape[0] * topk_idxs.shape[1]

        if args.src is not None:
            args.src.close()

        non_zero_idxs = attended_word_tensor.nonzero().view(-1)
        counts = attended_word_tensor[non_zero_idxs].tolist()
        toks = tokenizer.convert_ids_to_tokens(non_zero_idxs.tolist())
        attended_word_dict.update(zip(toks, counts))
    # end if of batch implementation
    print('Finished, total duration = {:.4}'.format(time.time() - total_start_time))

    top_attn_count = sorted(attended_word_dict.items(),key=lambda p: p[1],reverse=True)

    res_file = 'results/10000_attn_top15.txt'

    with open(res_file, 'w') as f:
        f.write('Total tokens attended: {}\n '.format(total_words_attended))
        for p in top_attn_count:
            f.write(p[0] + ' ' + str(p[1]) + '\n')

if __name__ == '__main__':
    main()
