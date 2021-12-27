# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

#reference: /remote-home/dmsong/FLAT/V12/bert_model

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.bert.configuration_bert import BertConfig

#from utils import get_crf_zero_init
from fastNLP import seq_len_to_mask


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len,max_seq_len+1, dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        n = torch.abs(n)
    else:
        n = torch.max(n, torch.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, word_embedding=None):
        super().__init__()
        self.config = config
        self.position_type = config.position_type
        self.position_fusion = config.position_fusion
        self.position_embedding = config.position_embedding
        self.hidden_size = config.hidden_size
        self.absolute_position_hidden_size = self.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id) \
            if word_embedding is None else word_embedding

        self.word_embedding_size = self.word_embeddings.embedding_dim
        self.word_embedding_liner = nn.Linear(self.word_embedding_size, self.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.position_type == "flat":

            relative_embedding = get_embedding(config.max_seq_len, self.hidden_size, rel_pos_init=config.rel_pos_init)
            pe_sum = relative_embedding.sum(dim=-1, keepdims=True)
            with torch.no_grad():
                relative_embedding = relative_embedding/pe_sum
            self.relative_position_embedding = nn.Embedding(
                config.max_seq_len * 2 + 1, self.hidden_size,
                _weight=relative_embedding,
            )
            self.relative_position_embedding.requires_grad_(False)
            self.pos_fusion_forward = nn.Sequential(nn.Linear(self.hidden_size * 4, self.hidden_size),
                                                nn.ReLU(inplace=True))
        elif self.position_type == "tupe":
            if self.position_embedding == "transformer":
                relative_embedding = get_embedding(config.max_seq_len, self.hidden_size, rel_pos_init=config.rel_pos_init)
                pe_sum = relative_embedding.sum(dim=-1, keepdims=True)
                with torch.no_grad():
                    relative_embedding = relative_embedding / pe_sum
                self.absolute_position_embedding = nn.Embedding(config.max_seq_len * 2 + 1, self.hidden_size,
                                                            _weight=relative_embedding,
                                                            )

            elif self.position_embedding == "bert":
                self.absolute_position_embedding = nn.Embedding(config.max_position_embeddings, config.bert_hidden_size)
                self.absolute_position_embedding.requires_grad_(False)
                self.absolute_position_embedding.load_state_dict(torch.load(config.position_embedding_path))
                self.absolute_position_hidden_size = config.bert_hidden_size
            else:
                raise NameError("please chose position_embedding in ['transformer', 'bert']")

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(self.absolute_position_hidden_size / config.num_attention_heads)
            self.pos_ln = nn.LayerNorm(self.absolute_position_hidden_size, eps=config.layer_norm_eps)
            self.pos_q_linear = nn.Linear(self.absolute_position_hidden_size, self.absolute_position_hidden_size)
            self.pos_k_linear = nn.Linear(self.absolute_position_hidden_size, self.absolute_position_hidden_size)
            self.pos_scaling = float(self.absolute_position_hidden_size / config.num_attention_heads * 2) ** -0.5

            self.absolute_position_embedding.requires_grad_(False)


            self.rel_pos_bins = 32
            self.max_rel_pos = 128
            self.bid_pos_bins = self.rel_pos_bins/2
            self.relative_position_embedding = nn.Embedding(self.rel_pos_bins + 1, self.num_attention_heads)
        elif self.position_type == "diet_abs":
            if self.position_embedding == "transformer":
                relative_embedding = get_embedding(config.max_seq_len, self.hidden_size,
                                                   rel_pos_init=config.rel_pos_init)
                pe_sum = relative_embedding.sum(dim=-1, keepdims=True)
                with torch.no_grad():
                    relative_embedding = relative_embedding / pe_sum
                self.absolute_position_embedding = nn.Embedding(config.max_seq_len * 2 + 1, self.hidden_size,
                                                                _weight=relative_embedding,
                                                                )

            elif self.position_embedding == "bert":
                self.absolute_position_embedding = nn.Embedding(config.max_position_embeddings, config.bert_hidden_size)
                self.absolute_position_embedding.requires_grad_(False)
                self.absolute_position_embedding.load_state_dict(torch.load(config.position_embedding_path))
                self.absolute_position_hidden_size = config.bert_hidden_size
            else:
                raise NameError("please chose position_embedding in ['transformer', 'bert']")

            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(self.absolute_position_hidden_size / config.num_attention_heads)
            self.pos_ln = nn.LayerNorm(self.absolute_position_hidden_size, eps=config.layer_norm_eps)
            self.pos_q_linear = nn.Linear(self.absolute_position_hidden_size, self.absolute_position_hidden_size)
            self.pos_k_linear = nn.Linear(self.absolute_position_hidden_size, self.absolute_position_hidden_size)
            self.pos_scaling = float(self.absolute_position_hidden_size / config.num_attention_heads * 2) ** -0.5

            self.absolute_position_embedding.requires_grad_(False)
        elif self.position_type == "diet_rel":
            self.rel_pos_bins = 32
            self.bid_pos_bins = self.rel_pos_bins/2
            self.relative_position_embedding = nn.Embedding(self.rel_pos_bins + 1, self.num_attention_heads)
        else:
            raise NameError("please chose position_type in ['flat', 'tupe', 'diet_abs', 'diet_rel', 'ours']")

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

    def get_rel_pos_bias(self, x):
        if self.rp_bucket.device != x.device:
            self.rp_bucket = self.rp_bucket.to(x.device)
        seq_len = x.size(1)
        rp_bucket = self.rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.relative_attention_bias.weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_ids, pos_s, pos_e):


        inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = self.word_embedding_liner(inputs_embeds)
        input_embeds = self.LayerNorm(inputs_embeds)
        input_embeds = self.dropout(input_embeds)

        batch = pos_s.size(0)
        max_seq_len = pos_s.size(1)

        if self.position_type == "flat":
            #print("inner max: ", torch.max(pos_s))
            #print(self.config.max_seq_len)
            #print(pos_s.unsqueeze(-1), pos_s.unsqueeze(-2))
            
            pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.config.max_seq_len
            pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.config.max_seq_len
            pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2) + self.config.max_seq_len
            pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2) + self.config.max_seq_len

            #print(pos_ss.shape)
            #print(torch.max(pos_ss.view(-1)))
            #print(max_seq_len)
            #print(self.relative_position_embedding)
            pos_ss = self.relative_position_embedding(pos_ss).view(size=[batch, max_seq_len, max_seq_len, -1])
            pos_se = self.relative_position_embedding(pos_se).view(size=[batch, max_seq_len, max_seq_len, -1])
            pos_es = self.relative_position_embedding(pos_es).view(size=[batch, max_seq_len, max_seq_len, -1])
            pos_ee = self.relative_position_embedding(pos_ee).view(size=[batch, max_seq_len, max_seq_len, -1])

            pos_4 = torch.cat([pos_ss, pos_se, pos_es, pos_ee], dim=-1)
            # pos_4 = torch.cat([pos_ss, pos_ee], dim=-1)
            rel_pos_embedding = self.pos_fusion_forward(pos_4)
            rel_pos_embedding = self.LayerNorm(rel_pos_embedding)
            rel_pos_embedding = self.dropout(rel_pos_embedding)

            return input_embeds, rel_pos_embedding, None
        elif self.position_type == "tupe":

            pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
            pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

            pos_mask = pos_ss > self.bid_pos_bins
            pos_ss = pos_ss.masked_fill(pos_mask, self.bid_pos_bins)
            pos_mask = pos_ss < -self.bid_pos_bins
            pos_ss = pos_ss.masked_fill(pos_mask, -self.bid_pos_bins)

            pos_mask = pos_ee > self.bid_pos_bins
            pos_ee = pos_ee.masked_fill(pos_mask, self.bid_pos_bins)
            pos_mask = pos_ee < -self.bid_pos_bins
            pos_ee = pos_ee.masked_fill(pos_mask, -self.bid_pos_bins)

            pos_ss = pos_ss + self.bid_pos_bins
            pos_ee = pos_ee + self.bid_pos_bins

            pos_ss = self.relative_position_embedding(pos_ss.int()).permute(0, 3, 1, 2)
            pos_ee = self.relative_position_embedding(pos_ee.int()).permute(0, 3, 1, 2)

            rel_pos_embedding = None
            if self.position_fusion == "ff_two":
                rel_pos_embedding = pos_ss + pos_ee
            elif self.position_fusion == "ff_four":
                pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
                pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)

                pos_mask = pos_se > self.bid_pos_bins
                pos_se = pos_se.masked_fill(pos_mask, self.bid_pos_bins)
                pos_mask = pos_se < -self.bid_pos_bins
                pos_se = pos_se.masked_fill(pos_mask, -self.bid_pos_bins)

                pos_mask = pos_es > self.bid_pos_bins
                pos_es = pos_es.masked_fill(pos_mask, self.bid_pos_bins)
                pos_mask = pos_es < -self.bid_pos_bins
                pos_es = pos_es.masked_fill(pos_mask, -self.bid_pos_bins)

                pos_se = pos_se + self.bid_pos_bins
                pos_es = pos_es + self.bid_pos_bins

                pos_se = self.relative_position_embedding(pos_se.int()).permute(0, 3, 1, 2)
                pos_es = self.relative_position_embedding(pos_es.int()).permute(0, 3, 1, 2)

                rel_pos_embedding = pos_ss + pos_se + pos_es + pos_ee
            else:
                raise NameError("please chose position_fusion in [ff_two, ff_four]")

            pos_s_embedding = self.absolute_position_embedding(pos_s)
            pos_e_embedding = self.absolute_position_embedding(pos_e)

            pos_s_embedding = self.pos_ln(pos_s_embedding)
            pos_e_embedding = self.pos_ln(pos_e_embedding)
            pos_s_embedding = self.dropout(pos_s_embedding)
            pos_e_embedding = self.dropout(pos_e_embedding)
            pos_s_q = self.pos_q_linear(pos_s_embedding) * self.pos_scaling
            pos_e_q = self.pos_q_linear(pos_e_embedding) * self.pos_scaling
            pos_s_k = self.pos_k_linear(pos_s_embedding)
            pos_e_k = self.pos_k_linear(pos_e_embedding)

            pos_s_q_layer = self.transpose_for_scores(pos_s_q)
            pos_e_q_layer = self.transpose_for_scores(pos_e_q)
            pos_s_k_layer = self.transpose_for_scores(pos_s_k)
            pos_e_k_layer = self.transpose_for_scores(pos_e_k)

            attention_s = torch.matmul(pos_s_q_layer, pos_s_k_layer.transpose(-1, -2))
            attention_e = torch.matmul(pos_e_q_layer, pos_e_k_layer.transpose(-1, -2))

            abs_pos_embedding = attention_s + attention_e

            return input_embeds, rel_pos_embedding, abs_pos_embedding
        elif self.position_type == "diet_abs":
            pos_s_embedding = self.absolute_position_embedding(pos_s)
            pos_e_embedding = self.absolute_position_embedding(pos_e)

            pos_s_embedding = self.pos_ln(pos_s_embedding)
            pos_e_embedding = self.pos_ln(pos_e_embedding)
            pos_s_embedding = self.dropout(pos_s_embedding)
            pos_e_embedding = self.dropout(pos_e_embedding)
            pos_s_q = self.pos_q_linear(pos_s_embedding) * self.pos_scaling
            pos_e_q = self.pos_q_linear(pos_e_embedding) * self.pos_scaling
            pos_s_k = self.pos_k_linear(pos_s_embedding)
            pos_e_k = self.pos_k_linear(pos_e_embedding)

            pos_s_q_layer = self.transpose_for_scores(pos_s_q)
            pos_e_q_layer = self.transpose_for_scores(pos_e_q)
            pos_s_k_layer = self.transpose_for_scores(pos_s_k)
            pos_e_k_layer = self.transpose_for_scores(pos_e_k)

            attention_s = torch.matmul(pos_s_q_layer, pos_s_k_layer.transpose(-1, -2))
            attention_e = torch.matmul(pos_e_q_layer, pos_e_k_layer.transpose(-1, -2))

            abs_pos_embedding = attention_s + attention_e

            return input_embeds, None, abs_pos_embedding
        elif self.position_type == "diet_rel":
            pos_ss = pos_s.unsqueeze(-1) - pos_s.unsqueeze(-2)
            pos_ee = pos_e.unsqueeze(-1) - pos_e.unsqueeze(-2)

            pos_mask = pos_ss > self.bid_pos_bins
            pos_ss = pos_ss.masked_fill(pos_mask, self.bid_pos_bins)
            pos_mask = pos_ss < -self.bid_pos_bins
            pos_ss = pos_ss.masked_fill(pos_mask, -self.bid_pos_bins)

            pos_mask = pos_ee > self.bid_pos_bins
            pos_ee = pos_ee.masked_fill(pos_mask, self.bid_pos_bins)
            pos_mask = pos_ee < -self.bid_pos_bins
            pos_ee = pos_ee.masked_fill(pos_mask, -self.bid_pos_bins)

            pos_ss = pos_ss + self.bid_pos_bins
            pos_ee = pos_ee + self.bid_pos_bins

            pos_ss = self.relative_position_embedding(pos_ss.int()).permute(0, 3, 1, 2)
            pos_ee = self.relative_position_embedding(pos_ee.int()).permute(0, 3, 1, 2)

            rel_pos_embedding = None
            if self.position_fusion == "ff_two":
                rel_pos_embedding = pos_ss + pos_ee
            elif self.position_fusion == "ff_four":
                pos_es = pos_e.unsqueeze(-1) - pos_s.unsqueeze(-2)
                pos_se = pos_s.unsqueeze(-1) - pos_e.unsqueeze(-2)

                pos_mask = pos_se > self.bid_pos_bins
                pos_se = pos_se.masked_fill(pos_mask, self.bid_pos_bins)
                pos_mask = pos_se < -self.bid_pos_bins
                pos_se = pos_se.masked_fill(pos_mask, -self.bid_pos_bins)

                pos_mask = pos_es > self.bid_pos_bins
                pos_es = pos_es.masked_fill(pos_mask, self.bid_pos_bins)
                pos_mask = pos_es < -self.bid_pos_bins
                pos_es = pos_es.masked_fill(pos_mask, -self.bid_pos_bins)

                pos_se = pos_se + self.bid_pos_bins
                pos_es = pos_es + self.bid_pos_bins

                pos_se = self.relative_position_embedding(pos_se.int()).permute(0, 3, 1, 2)
                pos_es = self.relative_position_embedding(pos_es.int()).permute(0, 3, 1, 2)

                rel_pos_embedding = pos_ss + pos_se + pos_es + pos_ee
            else:
                raise NameError("please chose position_fusion in [ff_two, ff_four]")
            return input_embeds, rel_pos_embedding, None
        else:
            raise NameError("please chose position_type in ['flat', 'tupe', 'diet']")


class BertSelfAttention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.position_type = config.position_type

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.relative = nn.Linear(config.hidden_size, self.all_head_size)

        self.u = nn.Parameter(torch.Tensor(self.num_attention_heads, self.attention_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_attention_heads, self.attention_head_size))
        self.u.data.normal_(mean=0.0, std=config.initializer_range)
        self.v.data.normal_(mean=0.0, std=config.initializer_range)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.layer = layer
        self.position_first_layer = config.position_first_layer

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        rel_embedding,
        abs_embedding,
        attention_mask,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        batch = mixed_key_layer.size(0)
        max_seq_len = mixed_key_layer.size(1)

        if self.position_type == "flat":

            rel_embedding = self.relative(rel_embedding)

            rel_embedding = torch.reshape(
                rel_embedding, [batch, max_seq_len, max_seq_len, self.num_attention_heads, self.attention_head_size]
            )

            # TODO 根据BERT源代码中的relative_position_scores修改这里
            u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
            query_layer_and_u = query_layer + u_for_c
            key_layer = key_layer.transpose(-1, -2)
            A_C = torch.matmul(query_layer_and_u, key_layer)

            query_for_b = query_layer.view([batch, self.num_attention_heads, max_seq_len, 1, self.attention_head_size])
            pos_embedding_for_b = rel_embedding.permute(0, 3, 1, 4, 2)

            query_for_b_and_v_for_d = query_for_b + self.v.view(1, self.num_attention_heads, 1, 1, self.attention_head_size)
            B_D = torch.matmul(query_for_b_and_v_for_d, pos_embedding_for_b).squeeze(-2)

            attention_scores = A_C + B_D

        elif self.position_type == "tupe":
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if self.position_first_layer:
                if self.layer == 0:
                    attention_scores = attention_scores + rel_embedding + abs_embedding
            else:
                attention_scores = attention_scores + rel_embedding + abs_embedding
        elif self.position_type == "diet_abs":
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if self.position_first_layer:
                if self.layer == 0:
                    attention_scores = attention_scores + abs_embedding
            else:
                attention_scores = attention_scores + abs_embedding
        elif self.position_type == "diet_rel":
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if self.position_first_layer:
                if self.layer == 0:
                    attention_scores = attention_scores + rel_embedding
            else:
                attention_scores = attention_scores + rel_embedding
        else:
            raise NameError("please chose position_type in ['flat', 'tupe', 'diet_abs', 'diet_rel', 'ours']")

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.self = BertSelfAttention(config, layer)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        rel_embedding,
        abs_embedding,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            rel_embedding,
            abs_embedding,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config, layer)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config, layer)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        rel_embedding,
        abs_embedding,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            rel_embedding,
            abs_embedding,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                rel_embedding,
                abs_embedding,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config, layer) for layer in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        rel_embedding,
        abs_embedding,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    rel_embedding,
                    abs_embedding,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    rel_embedding,
                    abs_embedding,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, word_embedding=None):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config, word_embedding)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids,
        pos_s,
        pos_e,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output,  rel_embedding, abs_embedding = self.embeddings(
            input_ids=input_ids, pos_s=pos_s, pos_e=pos_e
        )
        encoder_outputs = self.encoder(
            embedding_output,
            rel_embedding,
            abs_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertForFlat(BertPreTrainedModel):
    """
    dmsong's flat
    """
    def __init__(self, config, word_embedding):
        super(BertForFlat, self).__init__(config=config)
        self.bert = BertModel(config=config, add_pooling_layer=False, word_embedding=word_embedding)
        self.output = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vocab_size = config.num_labels 
        self.num_labels = config.num_labels
        #self.crf = get_crf_zero_init(config.num_labels)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

        self.use_crf = config.use_crf
        self.init_weights()

    def forward(
            self,
            input_ids,
            attention_mask,
            target,
            pos_s,
            pos_e,
            seq_len=None
    ):
        #print(type(input_ids))
        #print(input_ids)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            pos_s=pos_s,
            pos_e=pos_e
        )
        
        sequence_output = outputs[0]
        #max_seq_len = seq_len.max() if seq_len else 128
        last_hidden_state = self.dropout(sequence_output)
        #last_hidden_state = last_hidden_state[:, :max_seq_len, :]
        logits = self.output(last_hidden_state)[:, :target.shape[1], :]
        #print(logits.shape, target.shape)
        loss_fct = CrossEntropyLoss()
        #seq_len = torch.tensor([72])
        #print(self.vocab_size)
        #print(preds.shape, target.shape)
        #mask = seq_len_to_mask(seq_len).bool()
        if False:#self.use_crf:
            #if self.training:
            #    loss = self.crf(pred, target, mask).mean(dim=0)
            #    return {'loss': loss}
            #pred, _ = self.crf.viterbi_decode(pred, mask)
            #return {'pred': pred}
            return
        else:
            #activate_loss = mask.view(-1) == 1
            #active_logits = pred.view(-1, self.num_labels)
            #activate_labels = torch.where(
            #    activate_loss, target.view(-1), torch.tensor(loss_fct.ignore_index).type_as(target)
            #)
            #loss = loss_fct(active_logits[:activate_labels.shape[-1]], activate_labels)
            loss = loss_fct(logits.reshape(-1, self.vocab_size), target.view(-1))
            #if self.training:
            #    return loss
            #else:
            #    return loss, logits
            return loss, logits
