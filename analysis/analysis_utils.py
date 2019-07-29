import torch, logging
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from collections import defaultdict as DD


def prepare_models():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()
    mask_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    mask_model.eval()
    return tokenizer, model, mask_model


def cleanup(tokenizer, sent):
    """Cleans up text from gigaword for bert model input"""
    sent = sent.strip()
    sent = sent.replace('``', '\"')
    sent = sent.replace('\'\'', '\"')
    sent = sent.replace('-lrb-', '(')
    sent = sent.replace('-rrb-', ')')
    sent = sent.replace('<unk>', tokenizer.unk_token)
    sent = sent.replace('UNK', tokenizer.unk_token)
    sent = '[CLS] ' + sent.strip() + ' [SEP]'
    return sent

def batch_to_idx_tensor(tokenizer, text):
    """Converts a list of input text to input for bert model"""
    assert isinstance(text, list), 'Must input a list of strings!'

    str_tokens = [tokenizer.tokenize(cleanup(tokenizer, t)) for t in text]
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
    text = '[CLS] ' + cleanup(tokenizer, text) + ' [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    # print(tokenized_text)

    if masked_words is not None:
        for i, tok in enumerate(tokenized_text):
            if tok in masked_words:
                tokenized_text[i] = '[MASK]'

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    return tokens_tensor
