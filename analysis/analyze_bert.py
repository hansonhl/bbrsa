"""
Discover patterns in what BERT attends to. 'What does Bert look at?' paper shows
that there is a high entropy in beginning and ending layers' attention of BERT.
I would like to know about the distribution of words attended to by BERT at
these layers, if BERT were given news article sentences form the Gigaword
summary dataset.

In this script I look at attentions given a sentence.
I sum over the attentions over all heads and normalize it to get a
distribution, and record what words get the most attention.
"""


import torch, logging, argparse, pickle, time
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from collections import defaultdict as DD

from more_itertools import chunked
from analysis_utils import *



def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='''Script to find out what words BERT\'s
        attention attends to in layers 0, 9, 10, 11.''')
    parser.add_argument('-i', '--src', type=argparse.FileType('r'), metavar='PATH',
        help='''File containing multiple lines of input text, if not specified,
              uses some predefined text.''')
    parser.add_argument('-t', '--top', type=int, default=10, metavar='K',
        help='Find top K words that BERT attends to. Default 10.')
    parser.add_argument('-b', '--batch_size', type=int, default=20, metavar='B',
        help='Specify batch size=B. Default 20.')
    parser.add_argument('-g', '--gpu', action='store_true',
        help='Option to use GPU.')
    parser.add_argument('-a', '--all_layers', action='store_true',
        help='Output the attention of each layer')
    parser.add_argument('-T', '--out_top', type=int, metavar='N',
        help='Output top N words in final output. If -a is set, default value is 100')
    parser.add_argument('-m', '--mask', action='store_true',
        help='Mask attended words and compare predictions with original words. Not functional yet.')
    parser.add_argument('-o', '--out', type=argparse.FileType('w'),
        help='File to write results', required=True)
    args = parser.parse_args()

    top_k = args.top
    batch_size = args.batch_size
    use_gpu = args.gpu
    do_mask = args.mask
    all_layers = args.all_layers
    out_top_k = args.out_top

    print('all_layers', all_layers)

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

    if do_mask:
        mask_model = BertForMaskedLM.from_predtrained('bert-base-uncased')
        mask_model.eval()
        mask_token = tokenizer.mask_token
        mask_id = tokenizer.convert_tokens_to_ids([mask_token])[0]
        ignore_tokens = ['[CLS]', '[SEP]', '.', ',',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'said', '#', 'here', '<', 'news', '>', '`',
        'has', 'have', 'will', 'the', 'a', 'is', 'was', 'are',
        'on', 'as',  'after', 'this', 'in', 'with', 'to']
        ignore_ids = tokenizer.convert_tokens_to_ids(ignore_tokens)
        if use_gpu:
            mask_model = mask_model.cuda()

    if use_gpu:
        assert torch.cuda.is_available(), 'GPU unavailable!'
        model = model.cuda()
        print('-- Using GPU --')
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    total_words_attended = 0
    total_start_time = time.time()

    # if batch_size is None:
    #     count = 0
    #     layers = [0, 9, 10, 11]
    #     for line in src:
    #         if count % 100 == 0:
    #             print('Processed', count, 'lines')
    #         count += 1
    #         line = line.strip()
    #         tokens_tensor = str_to_idx_tensor(tokenizer, line)
    #         if use_gpu:
    #             tokens_tensor = tokens_tensor.cuda()
    #         str_tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist()[0])
    #         outputs = model(tokens_tensor)
    #
    #         cum_attn = []
    #
    #         for l in layers:
    #             layer = outputs[2][l]
    #             summed = layer.sum(dim=2).sum(dim=1).view(-1)
    #             summed = (summed / summed.sum(dim=0))
    #             cum_attn.append(summed)
    #
    #         all_attns = torch.stack(cum_attn).sum(dim=0)
    #         normalized_attn = (all_attns / all_attns.sum(dim=0)).tolist()
    #         sorted_by_attn = sorted(list(zip(normalized_attn, str_tokens)), key=lambda p: p[0], reverse=True)
    #
    #         for p in sorted_by_attn[:top_k]:
    #             attended_word_dict[p[1]] += 1
    #
    #         total_words_attended += len(sorted_by_attn[:top_k])
    #     if args.src is not None:
    #         args.src.close()

    # batch implementation
    if all_layers:
        attended_word_tensor = torch.zeros(12, tokenizer.vocab_size,
            device=device)
    else:
        layers = torch.tensor([0], device=device)
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

        attn = torch.stack(outputs[2])
        # attn has shape [num_layers (12), batch_size, num_heads, max_src_len, max_src_len]
        if all_layers:
            summed = attn.sum(dim=3).sum(dim=2) # [num_layers, batch_size, max_src_len]
            summed = summed / summed.sum(dim=2, keepdim=True) # normalize
            summed.transpose_(0,1)
            # summed has shape [bath_size, num_layers, max_src_len]
        else:
            attn = attn.index_select(0, layers)
            summed = attn.sum(dim=3).sum(dim=2).sum(dim=0).view(attn.shape[1], attn.shape[4])
            summed = summed / summed.sum(dim=1, keepdim=True) # normalize
            # summed has shape [batch_size, max_src_len]

        _, topk_idxs = summed.topk(top_k, sorted=True)
        # topk_idxs has shape [batch_size, top_k]

        # split by each item in batch
        split_batch_toks = batch_toks.split(1, dim=0)
        split_topk_idxs = topk_idxs.split(1, dim=0)

        for idxs, toks in zip(split_topk_idxs, split_batch_toks):
            attended_toks = toks.squeeze(0)[idxs.squeeze(0)]
            if all_layers:
                # record topk attended tokens for each layer
                for i in range(summed.shape[1]):
                    attended_word_tensor[i, attended_toks[i]] += 1
            else:
                attended_word_tensor[attended_toks] += 1

        total_words_attended += topk_idxs.shape[0] * topk_idxs.shape[1]

    if args.src is not None:
        args.src.close()

    if all_layers:
        f = args.out
        f.write('Total tokens attended: {}\n '.format(total_words_attended))
        for i in range(attended_word_tensor.shape[0]):
            non_zero_idxs = attended_word_tensor[i].nonzero().view(-1)
            counts = attended_word_tensor[i, non_zero_idxs].tolist()
            toks = tokenizer.convert_ids_to_tokens(non_zero_idxs.tolist())
            attended_word_dict = DD(int)
            attended_word_dict.update(zip(toks, counts))
            top_attn_count = sorted(attended_word_dict.items(),key=lambda p: p[1],reverse=True)

            out_top_k = 300 if not out_top_k else out_top_k
            f.write('\nLAYER {}\n'.format(i))
            for p in top_attn_count[:out_top_k]:
                f.write(p[0] + ' ' + str(p[1]) + '\n')
        f.close()
        print('Finished, total duration = {:.4}'.format(time.time() - total_start_time))
    else:
        non_zero_idxs = attended_word_tensor.nonzero().view(-1)
        counts = attended_word_tensor[non_zero_idxs].tolist()
        toks = tokenizer.convert_ids_to_tokens(non_zero_idxs.tolist())
        attended_word_dict = DD(int)
        attended_word_dict.update(zip(toks, counts))
        # end if of batch implementation
        print('Finished, total duration = {:.4}'.format(time.time() - total_start_time))

        top_attn_count = sorted(attended_word_dict.items(),key=lambda p: p[1],reverse=True)

        f = args.out
        f.write('Total tokens attended: {}\n '.format(total_words_attended))
        if out_top_k is not None:
            top_attn_count = top_attn_count[:out_top_k]
        for p in top_attn_count:
            f.write(p[0] + ' ' + str(p[1]) + '\n')
        f.close()

if __name__ == '__main__':
    main()
