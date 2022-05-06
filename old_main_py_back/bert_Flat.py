import os
import random
import time
import collections
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional,Dict, Union, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BertConfig,
    DataCollatorForSeq2Seq,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from transformers import trainer_utils, training_args
from transformers.trainer_pt_utils import nested_detach
from transformers.file_utils import (
    PaddingStrategy,
    is_sagemaker_mp_enabled
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from torch.cuda.amp import autocast
from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

from core import get_lattice_metrics, argument_init, get_lattice_dataset
from lib import subTrainer  
#from data.DatasetLoadingHelper import load_lattice_sighan
#from models.bart.modeling_bart_v2 import BartForConditionalGeneration
#from models.bert.modeling_bert_v2 import BertForFlat
from models.bert.modeling_flat_v1 import Lattice_Transformer_SeqLabel, BERT_SeqLabel 
from fastNLP.embeddings import BertEmbedding

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

logger = logging.getLogger(__name__)
    

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    dataset: str = field(default="sighan", metadata={"help":"dataset"})
    max_length: int = field(default=128, metadata={"help": "max length"})
    num_beams: int = field(default=4, metadata={"help": "num beams"})

@dataclass
class MyDataCollatorForSeq2Seq:
    """
    """
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):

        from copy import deepcopy

        f_copy = deepcopy(features)

        def simple_pad(f_copy, key):
            f_key = [ f[key] for f in f_copy ]
            if f_key is not None:
                max_length = max(len(l) for l in f_key)

                padding_side = "right"

                if key == "attention_mask":
                    label_pad_token_id = 0
                elif key in ["input_ids", "pos_s", "pos_e"]:
                    label_pad_token_id = 0
                elif key == "target":
                    label_pad_token_id= 0
                else:
                    label_pad_token_id = self.label_pad_token_id 

                for f in f_copy: 
                    remainder = [label_pad_token_id] * (max_length - len(f[key]))
                    f[key] = (
                        f[key] + remainder if padding_side == "right" else remainder + f[key]
                    )
            
            return f_copy

        for key in ["input_ids", "target", "pos_s", "pos_e"]:
            f_copy = simple_pad(f_copy, key)

        new = {}

        black_list = ["lattice", "attention_mask"]

        for key in f_copy[0].keys():
            if key not in black_list:    
                new[key] = []
        
        for feature in f_copy:
            for key in feature.keys():
                if key not in black_list:
                    new[key].append(feature[key])

        for key in new.keys():
            if key not in  black_list:
            #    new[key] = new[key]
                new[key] = torch.tensor(new[key]) 

        return new

class MyTrainer(subTrainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
 
        #loss = model(**inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss, _ = self.compute_loss(model, inputs)
        else:
            loss, _ = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            #with autograd.detect_anomaly():
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        #print(inputs)
        #print("compute loss")
        #exit()
        #print(inputs["input_ids"].shape)

        loss, logits = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        #if self.args.past_index >= 0:
        #    self._past = outputs[self.args.past_index]

        #if labels is not None:
        #    loss = self.label_smoother(outputs, labels)
        #else:
        #    # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return loss, logits

    def simple_pad_(self, x):
        f_copy = [ f for f in x ]
        if f_copy is not None:
            max_length = max(len(l) for l in f_copy)
            
            label_pad_token_id = 0

            padded = []

            for f in f_copy: 
                remainder = [label_pad_token_id] * (max_length - len(f))
                padded.append(
                    f + remainder
                )            
        return torch.tensor(padded)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.label_names = ["target"]
        #print(inputs.get("target") is not None) 
        has_labels = all(inputs.get(k) is not None for k in self.label_names )
        #has_labels = "target" in inputs.keys()

        #print(has_labels) 
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.

        #current_device = torch.cuda.current_device()

        #labels = inputs["target"]

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, logits = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                #if isinstance(outputs, dict):
                #    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                #else:
                #    logits = outputs[1:]
            else:
                #loss = None
                
                loss, logits = model(**inputs)
                #if isinstance(outputs, dict):
                #    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                #else:
                #    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                #if self.args.past_index >= 0:
                #    self._past = outputs[self.args.past_index - 1]

        #if prediction_loss_only:
        #    return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        #print(logits.shape)
        return (loss, torch.argmax(torch.softmax(logits, 2), -1), labels)
        #return loss, logits, labels


def run():
    # Args
    training_args = argument_init(MySeq2SeqTrainingArguments)

    set_seed(training_args.seed)

    # Tokenizer    
    #tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    #tokenizer = AutoTokenizer.from_pretrained(
    #    tokenizer_model_name_path
    #)

    # Dataset
    datasets, vocabs, embeddings = get_lattice_dataset(training_args.dataset) 

    # Model

    #print(vocabs['lattice'].padding_idx, vocabs["label"].padding_idx)

    max_seq_len = max(*map(lambda x: max(x['seq_len']), datasets.values()))
    #max_lattice_len = 640
    #config = BertConfig.from_pretrained(
    #    "hfl/chinese-roberta-wwm-ext"
    #)
    #config.num_labels = len(vocabs['label'])
    #config.vocab_size = len(vocabs['lattice'])
    #config.max_seq_len = 128#max_seq_len
    #config.max_lattice_len = 128#max_lattice_len
    #config.bert_hidden_size = 128#config.hidden_size
    #config.hidden_size = 128#args.hidden_size
    #config.num_attention_heads = 8
    #config.rel_pos_init = 1
    #config.learnable_position = False#args.learnable_position
    #config.num_hidden_layers = 1#args.num_hidden_layers
    #config.position_embedding_path = None#args.position_embedding_path
    #config.use_crf = False#args.use_crf
    #config.position_type = "flat"#args.position_type
    #config.position_embedding = ""#args.position_embedding
    #config.position_fusion = "four"#args.position_fusion
    #config.position_first_layer = False#args.position_first_layer

    mode = {}
    mode['debug'] = 0
    mode['gpumm'] = False

    dropout = collections.defaultdict(int)
    dropout['embed'] = 0.5
    dropout['gaz'] = 0.5
    dropout['output'] = 0.3
    dropout['pre'] = 0.5
    dropout['post'] = 0.3
    dropout['ff'] = 0.15
    dropout['ff_2'] = 0.15
    dropout['attn'] = 0

    device = torch.device('cuda')

    bert_embedding = BertEmbedding(vocabs['lattice'], model_dir_or_name='cn-wwm', requires_grad=False,
                                           word_dropout=0.01)

    #model = BertForFlat(config=config, word_embedding=embedding)
    #model = BERT_SeqLabel(bert_embedding,len(vocabs['label']),vocabs,after_bert="mlp")
    
    model = Lattice_Transformer_SeqLabel(
                                        lattice_embed=embeddings, bigram_embed=None, hidden_size=160,
                                        label_size=len(vocabs['label']),
                                        num_heads=8, num_layers=1, use_abs_pos=False, use_rel_pos=True,
                                        learnable_position=False, add_position=False,
                                        layer_preprocess_sequence='', layer_postprocess_sequence='an', ff_size=540, scaled=False, dropout=dropout, use_bigram=False,
                                        mode=mode, dvc=device, vocabs=vocabs,
                                        max_seq_len=max_seq_len,
                                        rel_pos_shared=True,
                                        k_proj=False,
                                        q_proj=True,
                                        v_proj=True,
                                        r_proj=True,
                                        self_supervised=False,
                                        attn_ff=False,
                                        pos_norm=False,
                                        ff_activate="relu",
                                        abs_pos_fusion_func="nonlinear_add",
                                        embed_dropout_pos='0',
                                        four_pos_shared=True,
                                        four_pos_fusion="ff_two",
                                        four_pos_fusion_shared=True,
                                        bert_embedding=bert_embedding
                                        ) 
    

    # Metrics
    compute_metrics = get_lattice_metrics()#tokenizer)

    # Data Collator

    data_collator = MyDataCollatorForSeq2Seq(
        #tokenizer=tokenizer,
        #model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=16
    )#my data collator  fix the length for bert.
    
    # Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,         
        train_dataset=datasets["train"],    
        eval_dataset=datasets["valid"],
        #tokenizer=tokenizer,
        data_collator=data_collator,      
        compute_metrics=compute_metrics#hint:num_beams and max_length effect heavily on metric["F1_score"], so I modify train_seq2seq.py to value default prediction_step function
    )

    # Train

    train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    
    metrics["train_samples"] = len(datasets["train"])

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation 
    # reference:https://github1s.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        metrics["eval_samples"] = len(datasets["valid"])

        trainer.log_metrics("eval", metrics)
        
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            datasets["test"], 
            metric_key_prefix="predict",
        )

        predictions = np.where(predict_results.predictions != -100, predict_results.predictions, 0)#remove the pad 

        def simple_decode(pred, vocab):
            return [vocab.to_word(i) for i in pred]

        predictions = [ "".join(simple_decode(pred, vocabs["label"])) for pred in  predictions ]

        index = 0
        for i in datasets['test']:
            predictions[index] = predictions[index][:( i["seq_len"] - i["lex_nums"]) ]
            index += 1

        metrics = predict_results.metrics

        metrics["predict_samples"] = len(datasets["test"])

        #print(torch.tensor(predictions).shape)

        #predictions = tokenizer.batch_decode(
        #        sequences=predictions, 
        #        skip_special_tokens=True, 
        #        clean_up_tokenization_spaces=True
        #)

        #predictions = [ "".join([i for i in pred  if '\u4e00' <= i <= '\u9fff']) for pred in predictions]

        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")

        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions))

    logger.info("*"*10 + "Curtain" + "*"*10)

    print("*"*10 + "over" + "*"*10)

    return

if __name__ == "__main__":
    run()
