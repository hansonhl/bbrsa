import torch, logging
from bbrsa.abstract_classes import BatchDistractor
from bbrsa.utils import idx_remap
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

class NextExampleDistractor(BatchDistractor):
    """Use next example in batch as distractor"""
    def __init__(self, batch_size, logger=None):
        super().__init__(batch_size, logger)
        self._d_factor = 2

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src):
        new_src = []
        for batch in _chunks(src, self.orig_batch_size):
            for i, x in enumerate(batch):
                new_src.append(x)
                next_id = 0 if i == len(batch) - 1 else i + 1
                new_src.append(batch[next_id])
        return new_src, self.new_batch_size

class IdenticalDistractor(BatchDistractor):
    """Use the sample itself as distractor"""
    def __init__(self, batch_size, logger=None):
        super().__init__(batch_size, logger)
        self._d_factor = 2

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src):
        new_src = []
        for x in src:
            new_src.append(x)
            new_src.append(x)
        return new_src, self.new_batch_size

class NextNDistractor(BatchDistractor):
    """Use next N examples in batch as distractor"""
    def __init__(self, batch_size, N, logger=None):
        assert batch_size >= N+1, 'Invalid N!'
        super().__init__(batch_size, logger)
        self._d_factor = N+1

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src):
        new_src = []
        for batch in _chunks(src, self.orig_batch_size):
            if len(batch) < self.d_factor:
                self._log('Dropping this batch that is too short', logging.WARNING)
                continue
            for i, x in enumerate(batch):
                ids = [j if j < len(batch) else j - len(batch) \
                    for j in range(i, i + self.d_factor)]
                new_src += [batch[j] for j in ids]

        return new_src, self.new_batch_size

class BertDistractor(BatchDistractor):
    """Change words that BERT attend to with BERT's own predictions"""
    def __init__(self, batch_size, logger=None):
        super().__init__(batch_size, logger)
        self._d_factor = 2

        # initialize bert
        self._log('Initializing Bert models for distractor', logging.INFO)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = self.tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased',
            output_attentions=True)
        self.model.eval()
        self.mask_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mask_model.eval()
        self._log('Finished initializing', logging.INFO)


        self.pad_token = tokenizer.pad_token
        self.pad_id = tokenizer.convert_tokens_to_ids([self.pad_token])[0]
        self.mask_token = tokenizer.mask_token
        self.mask_id = tokenizer.convert_tokens_to_ids([self.mask_token])[0]
        self.unk_token = tokenizer.unk_token
        self.unk_id = tokenizer.convert_tokens_to_ids([self.unk_token])[0]

        # insignificant tokens to filter out
        self.attn_top_k = 10
        self.ignore_toks = ['[CLS]', '[SEP]', self.unk_token]
        ignore_ids = tokenizer.convert_tokens_to_ids(self.ignore_toks)
        self.ignore_mask = torch.ones(tokenizer.vocab_size, dtype=torch.uint8)
        self.ignore_mask[ignore_ids] = 0

        self.generate_methods = ['layer0_attn', 'unmasked_surprisal']


    @property
    def d_factor(self):
        return self._d_factor

    # cleanup
    def _cleanup(self, sent):
        tokenizer = self.tokenizer
        sent = sent.strip()
        sent = sent.replace('``', '\"')
        sent = sent.replace('\'\'', '\"')
        sent = sent.replace('-lrb-', '(')
        sent = sent.replace('-rrb-', ')')
        sent = sent.replace('<unk>', self.unk_token)
        sent = sent.replace('UNK', self.unk_token)
        sent = '[CLS] ' + sent + ' [SEP]'
        return sent

    def _batch_to_idx_tensor(self, text):
        """Process batch a text for input to bert.

        Returns:
            (LongTensor, list[list[str]], LongTensor, LongTensor, list[list[None, int]]):

            * scrm_tok_id_tensor:  Tensor of token id's,
                ``(batch_size, max_src_len)``.
            * scrm_str_tokens: 2d list of tokens in string form,
                ``(batch_size, max_src_len)``.
            * scrm_attn_mask: Attention mask, non-zero where tok is not [PAD],
                ``(batch_size, max_src_len)``.
            * scrm_idxs: scramble idxs, scramble order applied to src text batch
                ``(batch_size,)``.
            * scrm_tok_remap: See output from :func:`_bert_token_remap()`
        """
        assert isinstance(text, list), 'Must input a list of strings!'
        tokenizer = self.tokenizer
        str_cleaned_up = [self._cleanup(t) for t in text]
        str_tokens = [tokenizer.tokenize(t) for t in str_cleaned_up]
        token_remap_idxs = [_bert_token_remap(s, t) for s, t in \
            zip(str_cleaned_up, str_tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(t) for t in str_tokens]
        seq_lens = torch.LongTensor(list(map(len, indexed_tokens)))
        tok_id_tensor = torch.zeros((len(indexed_tokens), seq_lens.max()), \
            dtype=torch.long)
        attn_mask_tensor = torch.zeros((len(indexed_tokens), seq_lens.max()), \
            dtype=torch.long)
        pad_idx = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        tok_id_tensor.fill_(pad_idx)

        for idx, (seq, seqlen) in enumerate(zip(indexed_tokens, seq_lens)):
            tok_id_tensor[idx, :seqlen] = torch.LongTensor(seq)
            attn_mask_tensor[idx, :seqlen] = 1

        scrm_seq_lens, scrm_idxs = seq_lens.sort(0, descending=True)
        scrm_tok_id_tensor = tok_id_tensor[scrm_idxs]
        scrm_str_tokens = [str_tokens[i.item()] for i in scrm_idxs]
        scrm_attn_mask = attn_mask_tensor[scrm_idxs]
        scrm_tok_remap = [token_remap_idxs[i.item()] for i in scrm_idxs]

        return scrm_tok_id_tensor, scrm_str_tokens, scrm_seq_lens, \
            scrm_attn_mask, scrm_idxs, scrm_tok_remap

    def generate(self, src, method='layer0_attn', ensure_different=True):
        """Replace top 5 words with highest attn in Bert to and get 1 distractor

        Args:
            src: list of strings
            ensure_different: ensures different word is replaced during distractor
                generation.

        Returns:
            list of input strings coupled with their distractors
        """
        assert method in self.generate_methods, 'invalid BERT distractor generation method!'

        layer0_attn = (method == 'layer0_attn')
        unm_surp = (method == 'unmasked_surprisal')

        tokenizer = self.tokenizer

        batch_tensor, str_toks, seq_lens, attn_mask, scrm_idxs, tok_remap \
            = self._batch_to_idx_tensor(src)

        if layer0_attn:
            with torch.no_grad():
                outputs = self.model(batch_tensor, attention_mask=attn_mask)

            # get layer 0 attention
            # attn has shape [batch_size, num_heads, max_src_len, max_src_len]
            attn = outputs[2][0]
            summed = attn.sum(dim=2).sum(dim=1) # [batch_size, max_src_len]
            summed = summed / summed.sum(dim=1, keepdim=True) # normalize

            _, topk_idxs = summed.topk(self.attn_top_k, sorted=True)
        elif unm_surp:
            with torch.no_grad():
                outputs = self.mask_model(batch_tensor, attention_mask=attn_mask)
            scores = outputs[0]
            log_probs = scores / scores.logsumexp(dim=2, keepdim=True)
            # scores and log_probs have shape [batch_size, max_src_len, vocab_size]
            r = torch.arange(log_probs.shape[1])
            topk_idxs = []

            for sent_log_probs, word_ids, seq_len in zip(log_probs, batch_tensor, seq_lens):
                surprisal = (-sent_log_probs[r, word_ids])[:seq_len]
                _, topk = surprisal.topk(self.attn_top_k, sorted=True)
                topk_idxs.append(topk)

        all_masked_tensors = []
        all_masked_idxs = []

        # get mask_id's and get new tensors with things masked out
        for attended_idxs, src_tok_ids in zip(topk_idxs, batch_tensor):
            # filter out insignificant words that are attended to,
            # e.g. [CLS] [SEP], see __init__ for more
            attended_toks = src_tok_ids[attended_idxs]
            signif_mask = self.ignore_mask[attended_toks]
            signif_idxs = attended_idxs[signif_mask]

            # mask top 5 words that are attended by bert
            masked_idxs = signif_idxs[:5]
            masked_toks = src_tok_ids.clone().detach()
            masked_toks[masked_idxs] = self.mask_id
            all_masked_tensors.append(masked_toks)
            all_masked_idxs.append(masked_idxs)

        masked_batch = torch.stack(all_masked_tensors)
        with torch.no_grad():
            mask_output = self.mask_model(masked_batch, attention_mask=attn_mask)

        all_distractors = []

        for pred, masked_idxs, src_tok_ids, remap_idxs, src_str in \
                zip(mask_output[0], all_masked_idxs, batch_tensor, tok_remap, src):
            # get top2 predictions for masked positions
            _, top2 = pred[masked_idxs].topk(2, dim=1)
            subs = top2[:,0]

            # ensure substitutions are different from orig toks
            if ensure_different:
                org_toks = src_tok_ids[masked_idxs]
                duplicate_mask = (subs == org_toks)
                subs[duplicate_mask] = top2[duplicate_mask, 1]

            # get substituted sentence
            sub_tok_ids = src_tok_ids.clone().detach()
            sub_tok_ids[masked_idxs] = subs
            sub_tok_ids = sub_tok_ids[sub_tok_ids != self.pad_id]
            sub_tok_strs = tokenizer.convert_ids_to_tokens(sub_tok_ids.tolist())
            all_distractors.append(_retokenize(src_str, sub_tok_strs, remap_idxs))

        # reorder and group together
        reorder_idxs = idx_remap(scrm_idxs)
        all_distractors = [all_distractors[i] for i in reorder_idxs]
        res = []
        for s, d in zip(src, all_distractors):
            res.append(s)
            res.append(d)
        return res


def _bert_token_remap(src, tgt):
    """Given src string and list of tokenizations by bert, get mapping idxs

    Args:
        src: string, must be processed by :func:`cleanup()`
        tgt: list of strings, must be output from bert tokenization of src.

    Returns:
        list of None or int, see below for example.

    e.g. 'foo is barring the bazes' => cleanup =>
         src = ['[CLS]', 'foo', 'is', 'barring', 'the', 'baz', "'s", 'house', '[SEP]']
         tgt = ['[CLS]', 'foo', 'is', 'barr', '##ing', 'the', 'baz', "'", 's', 'house' '[SEP]']
         output = [None, 0, 1, 2, 2, 3, 4, 5, 5, 6, None]
         use None for [CLS] and [SEP]
    """
    src_list = src.split() # assume already changed to lowercase
    src_i = 0
    cont = ''
    res = []
    for tgt_i, tok in enumerate(tgt):

        if tok.startswith('##') and len(tok) > 2:
            tok = tok.lstrip('##')
        cont += tok

        assert src_list[src_i].startswith(cont), \
            'Mismatch in src and tgt! curr={} src={}'\
            .format(curr, src_list[src_i])

        if cont == src_list[src_i]:
            if cont == '[CLS]' or cont == '[SEP]':
                res.append(None)
            else:
                res.append(src_i - 1)
            src_i += 1
            cont = ''
        else:
            res.append(src_i - 1)

    return res

def _retokenize(src_str, tgt, remap_idxs):
    src_list = src_str.split()
    res = []
    curr_i = 0
    cont = ''
    for tok_i, (tok, idx) in enumerate(zip(tgt, remap_idxs)):
        if tok == '[CLS]' or tok == '[SEP]':
            continue
        if tok.startswith('##') and len(tok) > 2:
            tok = tok.lstrip('##')
        cont += tok
        if remap_idxs[tok_i + 1] != curr_i:
            if cont == '(' or cont == ')' or cont == '\"' or cont == '[UNK]':
                cont = src_list[curr_i]
            res.append(cont)
            cont = ''
            curr_i += 1
    return ' '.join(res)


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]
