import os
import random
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional,Dict, Union, Any, Tuple, List

import numpy as np
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import transformers
from transformers import (
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
from transformers import BertForMaskedLM
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.training_args import TrainingArguments

from core import get_super_magic_dataset, get_magic_expand_dataset, get_magic_plus_dataset, get_magic_lang8_dataset, get_magic_dataset, get_metrics, argument_init
from lib import subTrainer  
from models.bert.modeling_bert_v3 import BertForMaskedLM_v2 
#from transformers import BertForMaskedLM

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
        """
        """
        from copy import deepcopy

        f_copy = deepcopy(features)

        shared_max_length = max([ len(i['input_ids']) for i in f_copy])


        for i in range(len(f_copy)):
            f_copy[i]["raw_length"] = []

        for i in range(len(f_copy)):
            f_copy[i]["raw_length"].append(len(f_copy[i]["input_ids"]))

        def simple_pad(f_copy, key):
            f_key = [ f[key] for f in f_copy ]
            if f_key is not None:
                max_length = max(len(l) for l in f_key)

                padding_side = "right"

                if key == "attention_mask":
                    label_pad_token_id = 0
                elif key in ["input_ids", "lattice"]:
                    label_pad_token_id = 0
                elif key == "labels":
                    max_length = shared_max_length
                    label_pad_token_id= -100
                else:
                    label_pad_token_id = self.label_pad_token_id 

                for f in f_copy: 
                    remainder = [label_pad_token_id] * (max_length - len(f[key]))
                    f[key] = (
                        f[key] + remainder if padding_side == "right" else remainder + f[key]
                    )
            
            return f_copy

        for key in ["input_ids", "lattice", "labels", "attention_mask"]:
            f_copy = simple_pad(f_copy, key)

        new = {}

        black_list = []

        for key in f_copy[0].keys():
            if key not in black_list:    
                new[key] = []
        
        for feature in f_copy:
            for key in feature.keys():
                if key not in black_list:
                    new[key].append(feature[key])

        for key in new.keys():
            if key not in  black_list:
                #print(key)
            #    new[key] = new[key]
                new[key] = torch.tensor(new[key]) 

        #lets random mask
        for i in range(len(new["input_ids"])):
            #print(new["sub_length"][i], len(new["input_ids"][i]), new["sub_length"][i], len( new["raw_length"][0] ))

            left = torch.ones(new["sub_length"][i])

            right = torch.where( torch.randn( new["raw_length"][i] - new["sub_length"][i] ) > 0.6 , 1, 0)

            pad = torch.zeros( len(new["input_ids"][i]) - left.shape[0] - right.shape[0] )

            mask = torch.cat( (left, right, pad) , dim=0)
            
            new["input_ids"][i] = new["input_ids"][i] * mask

        new.pop("raw_length")

        return new

class MyTrainer(subTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                
                outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)


        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, torch.argmax(torch.softmax(logits, 2), -1), labels)
        #return loss, logits, labels


def run():
    # Args
    training_args = argument_init(MySeq2SeqTrainingArguments)

    set_seed(training_args.seed)

    # Tokenizer    
    tokenizer_model_name_path="hfl/chinese-roberta-wwm-ext"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name_path
    )

    # Dataset
    #train_dataset, eval_dataset, test_dataset = get_magic_dataset(training_args.dataset, ".")  
    train_dataset, eval_dataset, test_dataset = get_magic_plus_dataset(training_args.dataset, ".")
    #train_dataset, eval_dataset, test_dataset, tokenizer = get_super_magic_dataset(training_args.dataset, ".") #balanced within target and special token <RAW>
    
    # Model
    model = BertForMaskedLM_v2.from_pretrained(
        "hfl/chinese-roberta-wwm-ext"#"bert-base-chinese"#
        #"./tmp/sighan/bert_MaksedLM_base_raw_v2.epoch10.bs128"
    ) #base

    #model.resize_token_embeddings(len(tokenizer))

    # Metrics
    compute_metrics = get_metrics()

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
        train_dataset=train_dataset,    
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,      
        compute_metrics=compute_metrics#hint:num_beams and max_length effect heavily on metric["F1_score"], so I modify train_seq2seq.py to value default prediction_step function
    )

    # Train

    train_result = trainer.train()

    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation 
    # reference:https://github1s.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate(metric_key_prefix="eval")
        
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, 
            metric_key_prefix="predict",
        )

        predictions = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)#remove the pad 

        metrics = predict_results.metrics

        metrics["predict_samples"] = len(test_dataset)

        #print(torch.tensor(predictions).shape)

        predictions = tokenizer.batch_decode(
                sequences=predictions, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
        )

        predictions = [pred.strip() for pred in predictions]

        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")

        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(predictions))

    logger.info("*"*10 + "Curtain" + "*"*10)

    print("*"*10 + "over" + "*"*10)

    return

if __name__ == "__main__":
    run()
