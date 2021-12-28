
import os
import random
import time
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional,Dict, Union, Any, Tuple, List

import nltk
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
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers import Trainer, Seq2SeqTrainer
from transformers import TrainingArguments
from transformers.trainer_pt_utils import nested_detach
from transformers import trainer_utils, training_args

from core import get_dataset, get_metrics, argument_init
from lib import subTrainer, FoolDataCollatorForSeq2Seq 
from data.DatasetLoadingHelper import load_ctc2021, load_sighan
from models.bart.modeling_bart_v2 import BartForMaskedLM

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

logger = logging.getLogger(__name__)


@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    dataset: str = field(default="sighan", metadata={"help":"dataset"})
    max_length: int = field(default=128, metadata={"help": "max length"})
    num_beams: int = field(default=4, metadata={"help": "num beams"})

class MyTrainer(subTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
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
    train_dataset, eval_dataset, test_dataset = get_dataset(training_args.dataset) 

    # Model
    model = BartForMaskedLM.from_pretrained(
        #'/remote-home/share/yfshao/bart-zh/arch12-2-new-iter8w/'
        '/remote-home/xtzhang/CTC/CTC2021/SpecialEdition/models/bart/bart-zh/arch12-2-new-iter8w'
    ) #base

    # Metrics
    compute_metrics = get_metrics()

    # Data Collator
    data_collator = FoolDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None
    )
    
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

    max_length = 128
    
    num_beams =  4

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate(
            max_length=training_args.max_length, 
            num_beams=training_args.num_beams, 
            metric_key_prefix="eval"
            )
        
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset, 
            metric_key_prefix="predict", 
            max_length=max_length, 
            num_beams=num_beams
        )

        predict_results = trainer.predict(
            test_dataset, 
            metric_key_prefix="predict",
        )

        predictions = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)#remove the pad 

        metrics = predict_results.metrics

        metrics["predict_samples"] = len(test_dataset)

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

    return

if __name__ == "__main__":
    run()
