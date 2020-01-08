import torch, logging
from bbrsa.abstract_classes import BatchDistractor
from bbrsa.utils import idx_remap, chunks
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM

class AsIsDistractor(BatchDistractor):
    """Use distractors as they are already arranged in the batch"""
    def __init__(self, d_factor, logger=None):
        super().__init__(logger)
        self._d_factor = d_factor

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src, opts=None):
        self.orig_batch_size = opts.batch_size
        return src, self.new_batch_size

class NextExampleDistractor(BatchDistractor):
    """Use next example in batch as distractor"""
    def __init__(self, opts, logger=None):
        super().__init__(logger)
        self._d_factor = 2

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src, opts):
        self.orig_batch_size = opts.batch_size
        new_src = []
        for batch in chunks(src, self.orig_batch_size):
            for i, x in enumerate(batch):
                new_src.append(x)
                next_id = 0 if i == len(batch) - 1 else i + 1
                new_src.append(batch[next_id])
        return new_src, self.new_batch_size

class IdenticalDistractor(BatchDistractor):
    """Use the sample itself as distractor"""
    def __init__(self, opts, logger=None):
        super().__init__(logger)
        self._d_factor = 2

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src, opts):
        self.orig_batch_size = opts.batch_size
        new_src = []
        for x in src:
            new_src.append(x)
            new_src.append(x)
        return new_src, self.new_batch_size

class NextNDistractor(BatchDistractor):
    """Use next N examples in batch as distractor"""
    def __init__(self, opts, logger=None):
        super().__init__(logger)
        self._d_factor = 4

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src, opts):
        N = opts.nextn_distr_N
        assert opts.batch_size >= N+1, 'Invalid N!'
        self._d_factor = N+1
        self.orig_batch_size = opts.batch_size

        new_src = []
        for batch in chunks(src, self.orig_batch_size):
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
    def __init__(self, opts, logger=None):
        super().__init__(logger)
        self._d_factor = 2
        self.gpu = False # opts.gpu
        self.device = torch.device('cuda') if self.gpu else torch.device('cpu')

        # initialize bert
        logging.basicConfig(level=logging.WARNING)
        self._info('>> Initializing Bert models for distractor')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer = self.tokenizer
        self.model = BertModel.from_pretrained('bert-base-uncased',
            output_attentions=True)
        self.model.eval()
        self.mask_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mask_model.eval()
        # if self.gpu:
        #     self.model = self.model.cuda()
        #     self.mask_model = self.mask_model.cuda()
        self._info('>> Finished initializing')

        self.pad_token = tokenizer.pad_token
        self.pad_id = tokenizer.convert_tokens_to_ids([self.pad_token])[0]
        self.mask_token = tokenizer.mask_token
        self.mask_id = tokenizer.convert_tokens_to_ids([self.mask_token])[0]
        self.unk_token = tokenizer.unk_token
        self.unk_id = tokenizer.convert_tokens_to_ids([self.unk_token])[0]

        # insignificant tokens to filter out
        excl0 = ['[CLS]', '[SEP]', self.unk_token]
        excl1 = ['[CLS]', '[SEP]', self.unk_token,'(', ')',
                 '\'', '\"', '.', ',', '#', '!', '?', '*',]
        excl2 = excl1 + ['monday', 'tuesday', 'wednesday', 'thursday',
                         'friday', 'saturday', 'sunday']
        excl3 = excl2 + ['the', 'a', 'an', 'some', 'of',
                         'have', 'had', 'has',
                         'say', 'said', 'says',
                         'be', 'been', 'being', 'is', 'are', 'was', 'were',
                         'this', 'that', 'these', 'there']

        self.exclusion_set = [excl0, excl1, excl2, excl3]
        self.ignore_toks = self.exclusion_set[opts.bert_distr_exclusion_set]
        self._debug('Ignore_toks:')
        self._debug(' '.join(self.ignore_toks))

        ignore_ids = tokenizer.convert_tokens_to_ids(self.ignore_toks)
        self.ignore_mask = torch.ones(tokenizer.vocab_size, dtype=torch.bool,
                                      device=self.device)
        self.ignore_mask[ignore_ids] = False

        if opts.bert_distr_no_subword_repl:
            for k, v in tokenizer.vocab.items():
                if k.startswith('##'):
                    self.ignore_mask[v] = False
                    self.ignore_toks.append(k)

        self.generate_methods = ['layer0_attn', 'unmasked_surprisal']


    @property
    def d_factor(self):
        return self._d_factor

    # cleanup
    def _cleanup(self, sent):
        tokenizer = self.tokenizer
        sent = sent.strip()
        sent = sent.replace('`', '\'')
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

        All outputs are sorted (i.e. 'scrambled') according to sequence length.

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
        seq_lens = torch.tensor(list(map(len, indexed_tokens)),
            dtype=torch.long, device=self.device)
        tok_id_tensor = torch.zeros((len(indexed_tokens), seq_lens.max()),
            dtype=torch.long, device=self.device)
        attn_mask_tensor = torch.zeros((len(indexed_tokens), seq_lens.max()),
            dtype=torch.bool, device=self.device)
        pad_idx = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        tok_id_tensor.fill_(pad_idx)

        for idx, (seq, seqlen) in enumerate(zip(indexed_tokens, seq_lens)):
            tok_id_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long,
                device=self.device)
            attn_mask_tensor[idx, :seqlen] = True

        scrm_seq_lens, scrm_idxs = seq_lens.sort(0, descending=True)
        scrm_tok_id_tensor = tok_id_tensor[scrm_idxs]
        scrm_str_tokens = [str_tokens[i] for i in scrm_idxs]
        scrm_attn_mask = attn_mask_tensor[scrm_idxs]
        scrm_tok_remap = [token_remap_idxs[i] for i in scrm_idxs]

        return scrm_tok_id_tensor, scrm_str_tokens, scrm_seq_lens, \
            scrm_attn_mask, scrm_idxs, scrm_tok_remap

    def _generate_batch(self, src, opts):
        method = opts.bert_distr_method
        self._d_factor = opts.bert_distr_d_factor
        num_distractors = opts.bert_distr_d_factor - 1
        salient_topk = opts.bert_distr_salient_topk
        mask_topk = opts.bert_distr_mask_topk
        repl_search_top = opts.bert_distr_repl_search_top
        repl_search_bottom = opts.bert_distr_repl_search_bottom
        ensure_different = opts.bert_distr_ensure_different

        layer0_attn = (method == 'layer0_attn')
        unm_surp = (method == 'unmasked_surprisal')

        tokenizer = self.tokenizer
        batch_tensor, str_toks, seq_lens, attn_mask, scrm_idxs, tok_remap \
            = self._batch_to_idx_tensor(src)

        if layer0_attn:
            raise NotImplementedError
            # with torch.no_grad():
            #     outputs = self.model(batch_tensor, attention_mask=attn_mask)
            #
            # # get layer 0 attention
            # # attn has shape [batch_size, num_heads, max_src_len, max_src_len]
            # attn = outputs[2][0]
            # summed = attn.sum(dim=2).sum(dim=1) # [batch_size, max_src_len]
            # summed = summed / summed.sum(dim=1, keepdim=True) # normalize
            #
            # _, topk_idxs = summed.topk(salient_topk, sorted=True)
        elif unm_surp:
            with torch.no_grad():
                outputs = self.mask_model(batch_tensor, attention_mask=attn_mask)
            scores = outputs[0]
            log_probs = scores / scores.logsumexp(dim=2, keepdim=True)
            # scores and log_probs have shape [batch_size, max_src_len, vocab_size]
            r = torch.arange(log_probs.shape[1], device=self.device)
            topk_idxs = []

            for sent_log_probs, word_ids, seq_len in zip(log_probs, batch_tensor, seq_lens):
                surprisal = -(sent_log_probs[r, word_ids])[:seq_len]

                # filter out insignificant words that are attended to,
                # e.g. [CLS] [SEP], see __init__ for more
                keep_mask = self.ignore_mask[word_ids[:seq_len]]
                surprisal[torch.bitwise_not(keep_mask)] = -float('inf')

                _, topk = surprisal.topk(mask_topk, sorted=True)
                topk_idxs.append(topk)

        all_masked_tensors = []
        all_masked_idxs = []

        # get mask_id's and get new tensors with things masked out
        for masked_idxs, src_tok_ids in zip(topk_idxs, batch_tensor):
            masked_toks = src_tok_ids.clone().detach()
            masked_toks[masked_idxs] = self.mask_id
            all_masked_tensors.append(masked_toks)
            all_masked_idxs.append(masked_idxs)

        masked_batch = torch.stack(all_masked_tensors)
        with torch.no_grad():
            mask_output = self.mask_model(masked_batch, attention_mask=attn_mask)

        all_distr_sets = []
        scrm_src = [src[i] for i in scrm_idxs]
        for pred, masked_idxs, src_tok_ids, remap_idxs, src_str in \
            zip(mask_output[0], all_masked_idxs, batch_tensor, tok_remap, scrm_src):
            # get predictions for masked positions

            # _, repl_candidates = pred[masked_idxs].topk(repl_search_bottom, dim=1)
            _, sorted_candidates = pred[masked_idxs].sort(dim=1, descending=True)
            repl_candidates = sorted_candidates[:, repl_search_top:repl_search_bottom]
            num_candidates = repl_candidates.shape[1]
            num_positions = repl_candidates.shape[0]

            repl_idxs = torch.randint(num_candidates, (num_distractors, num_positions),
                                      device=self.device)

            rng = torch.arange(num_positions, device=self.device)
            subs = repl_candidates[rng, repl_idxs]
            # if repl_idxs is has shape [num_positions , num_distractors]
            # then repl_candidates[rng, repl_idxs] returns [num_positions, num_distractors]

            # ensure substitutions are different from orig toks
            if ensure_different:
                org_toks = src_tok_ids[masked_idxs]
                duplicate_mask = (subs == org_toks)
                new_repl_idxs = repl_idxs.clone().detach()
                new_repl_idxs[duplicate_mask] += 1
                new_repl_idxs[new_repl_idxs >= num_candidates] = 0
                replacements = repl_candidates[rng, new_repl_idxs]

            # for each sentence in batch, multiple distractors form a set
            distr_set = []
            for repl in subs:
                distr_sent = src_tok_ids.clone().detach()
                distr_sent[masked_idxs] = repl
                distr_sent = distr_sent[distr_sent != self.pad_id]
                distr_tok_strs = tokenizer.convert_ids_to_tokens(distr_sent.tolist())
                distr_set.append(_retokenize(src_str, distr_tok_strs, remap_idxs))
            all_distr_sets.append(distr_set)

        # reorder and group together
        reorder_idxs = idx_remap(scrm_idxs)
        reorder_distr_sets = [all_distr_sets[i] for i in reorder_idxs]
        res = []
        for s, d_set in zip(src, reorder_distr_sets):
            res.append(s)
            res += d_set
        return res


    def generate(self, src, opts):
        """Replace top 5 salient words in Bert to get 1 distractor

        Measure of salience is determined by `method` parameter:

        `layer0_attn` uses the total attention on each word in the source
        article.

        `unmasked_surprisal` uses -log(prob(word)) for each position in the
        source given by the BERT language model, with the source as input.

        Args:
            src: list of strings
            opts: options

        Returns:
            list of input strings coupled with their distractors
        """
        self.orig_batch_size = 32

        assert opts.bert_distr_method in self.generate_methods, \
            'invalid BERT distractor generation method!'

        res = []
        tot_batches = len(src) // self.orig_batch_size

        for i, batch in enumerate(chunks(src, self.orig_batch_size)):
            if i % 10 == 9:
                self._info('>> BertDistractor batch {} / {}'\
                           .format(i+1, tot_batches))
            res += self._generate_batch(batch, opts)

        return res, self.new_batch_size


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
