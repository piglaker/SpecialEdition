
from torch import nn
from transformers import BertForMaskedLM, BertModel
from fastNLP import seq_len_to_mask
import torch
import torch.nn.functional as F
from .args import ARGS


class DoubleBertCSC(nn.Module):
    def __init__(self, model_name, bert_type, cl_weight=0.03, only_diff=False,
                 no_cl=False, layer_idx=-2):
        super(DoubleBertCSC, self).__init__()
        # self.fix_bert = BertModel.from_pretrained(model_name)
        if bert_type == 'mlm':
            self.bert = BertForMaskedLM.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        self.cl_weight = cl_weight
        self.only_diff = only_diff
        self.no_cl = no_cl
        self.layer_idx = layer_idx

    def forward(self, src_tokens, tgt_tokens, src_seq_len, no_target_mask, not_equal):
        no_target_mask = no_target_mask.bool()
        attention_mask = seq_len_to_mask(src_seq_len)
        bert_outputs = self.bert(src_tokens, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)

        bert_last_hidden_state = bert_outputs.hidden_states[self.layer_idx]
        logits = self._get_logits(bert_outputs)

        # with torch.no_grad():
        # self.fix_bert.eval()
        fix_outputs = self.bert(tgt_tokens, attention_mask, output_hidden_states=True, return_dict=True)
        fix_last_hidden_state = fix_outputs.hidden_states[self.layer_idx].detach()

        loss = F.cross_entropy(logits.transpose(1, 2), tgt_tokens.masked_fill(no_target_mask, -100), reduction='mean')
        if not self.no_cl:
            if self.cl_weight>0:
                if ARGS.get('pre_norm', False):
                    fix_last_hidden_state = F.normalize(fix_last_hidden_state, dim=-1)
                    bert_last_hidden_state = F.normalize(bert_last_hidden_state, dim=-1)

                if self.only_diff==1:
                    loss2 = torch.norm(fix_last_hidden_state - bert_last_hidden_state, dim=-1).masked_fill(not_equal.eq(0), 0).sum()/not_equal.sum()
                elif self.only_diff==0:
                    loss2 = torch.norm(fix_last_hidden_state - bert_last_hidden_state, dim=-1).masked_fill(no_target_mask, 0).sum()/no_target_mask.eq(0).sum()
                elif self.only_diff==2:
                    loss2 = torch.norm(fix_last_hidden_state - bert_last_hidden_state, dim=-1).masked_fill(attention_mask.eq(0), 0).sum()/attention_mask.sum()
                elif self.only_diff==3:
                    not_equal = src_tokens.ne(tgt_tokens)
                    equal = torch.logical_and(src_tokens.eq(tgt_tokens), attention_mask)
                    loss2_1 = torch.norm(fix_last_hidden_state - bert_last_hidden_state, dim=-1).masked_fill(not_equal.eq(0), 0).sum() / not_equal.sum()
                    loss2_2 = torch.norm(fix_last_hidden_state - bert_last_hidden_state, dim=-1).masked_fill(equal.eq(0), 0).sum() / equal.sum()

                    loss2 = loss2_1 * ARGS.get('neq_weight', 0.1) + loss2_2
                elif self.only_diff == 5:
                    loss2 = (1 - (fix_last_hidden_state*bert_last_hidden_state).sum(dim=-1).masked_fill(attention_mask.eq(0), 0).sum())/attention_mask.sum()
                elif self.only_diff==4:
                    scores = torch.einsum('blh,bkh->blk', bert_last_hidden_state, fix_last_hidden_state)  # bsz x len x len
                    scores = scores.masked_fill(attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1), 0)
                    scores = torch.exp(scores / 2)
                    pos_scores = torch.diagonal(scores, dim1=1, dim2=2)  # bsz x l
                    neg_scores = scores.sum(dim=-1)
                    loss2 = -(pos_scores / neg_scores).log().masked_fill(attention_mask.eq(0), 0).sum() / attention_mask.sum()
                # loss2 = 1 - ((fix_last_hidden_state*bert_last_hidden_state).sum(dim=-1)/torch.norm(fix_last_hidden_state, dim=-1)\
                #         /torch.norm(bert_last_hidden_state, dim=-1)).masked_fill(not_equal.eq(0), 0)
                # loss2 = loss2.sum()/not_equal.sum()

                loss = loss + self.cl_weight*loss2

            if ARGS.get('gold_loss_weight', 0)>0:
                loss3 = F.cross_entropy(fix_outputs.logits.transpose(1, 2), tgt_tokens.masked_fill(no_target_mask, -100), reduction='mean')
                loss = loss + ARGS.get('gold_loss_weight', 1)*loss3

        return {'loss': loss}

    def predict(self, src_tokens, src_seq_len):
        attention_mask = seq_len_to_mask(src_seq_len)
        bert_outputs = self.bert(src_tokens, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        logits = self._get_logits(bert_outputs)

        pred = logits.argmax(dim=-1)
        return {'pred': pred}

    def _get_logits(self, bert_outputs):
        if hasattr(bert_outputs, 'logits'):
            return bert_outputs.logits
        bert_last_hidden_state = bert_outputs.hidden_states[-1]
        logits = torch.einsum('blh,vh->blv', bert_last_hidden_state, self.bert.embeddings.word_embeddings.weight)
        return logits


class TestBertCSC(nn.Module):
    def __init__(self, model_name, bert_type, cl_weight=0.03):
        super(TestBertCSC, self).__init__()
        if bert_type == 'mlm':
            self.bert = BertForMaskedLM.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        self.cl_weight = cl_weight

    def forward(self, src_tokens, tgt_tokens, src_seq_len, no_target_mask, not_equal):
        not_equal = not_equal.bool()
        src_tokens = src_tokens.masked_fill(not_equal, 103)
        attention_mask = seq_len_to_mask(src_seq_len)
        bert_outputs = self.bert(src_tokens, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        logits = self._get_logits(bert_outputs)

        pred = logits.argmax(dim=-1)
        return {'pred': pred}

    def _get_logits(self, bert_outputs):
        if hasattr(bert_outputs, 'logits'):
            return bert_outputs.logits
        bert_last_hidden_state = bert_outputs.hidden_states[-1]
        logits = torch.einsum('blh,vh->blv', bert_last_hidden_state, self.bert.embeddings.word_embeddings.weight)
        return logits