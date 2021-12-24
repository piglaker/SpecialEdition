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

from core import (
    get_dataset, 
    get_metrics, 
    argument_init, 
    get_ReaLiSe_dataset,
)
from lib import subTrainer  
from data.DatasetLoadingHelper import load_ctc2021, load_sighan
#from models.bart.modeling_bart_v2 import BartForConditionalGeneration
from transformers import BertForMaskedLM

#os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

logger = logging.getLogger(__name__)
    

@dataclass
class MySeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    dataset: str = field(default="sighan", metadata={"help":"dataset"})
    max_length: int = field(default=128, metadata={"help": "max length"})
    num_beams: int = field(default=4, metadata={"help": "num beams"})

@dataclass
class MyDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            if max_label_length is not None and self.pad_to_multiple_of is not None and (max_label_length % self.pad_to_multiple_of != 0):
                max_label_length = ((max_label_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        print(features)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

@dataclass
class FoolDataCollatorForSeq2Seq:
    """
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
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

        def simple_pad(f_copy, key):
            f_key = [ f[key] for f in f_copy ]
            if f_key is not None:
                max_length = max(len(l) for l in f_key)

                padding_side = "right"

                if key == "attention_mask":
                    label_pad_token_id = 0
                elif key in ["input_ids"]:
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

        for key in ["input_ids", "labels", "attention_mask"]:
            f_copy = simple_pad(f_copy, key)

        new = {}

        black_list = []

        for key in f_copy[0].keys():  
            new[key] = []
        
        for feature in f_copy:
            for key in feature.keys():
                new[key].append(feature[key])

        for key in new.keys():
            new[key] = torch.tensor(new[key]) 

        return new

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
    train_dataset, eval_dataset, test_dataset = get_ReaLiSe_dataset()#get_dataset(training_args.dataset) 

    # Model
    model = BertForMaskedLM.from_pretrained(
        "hfl/chinese-roberta-wwm-ext"#"bert-base-chinese"
    ) #base

    # Metrics
    compute_metrics = get_metrics()

    # Data Collator

    data_collator = FoolDataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=64
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
