from datasets import load_dataset
from transformers import  T5Tokenizer , DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration,Seq2SeqTrainingArguments,Seq2SeqTrainer,AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer , T5Tokenizer , DataCollatorForSeq2Seq
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import os
from peft import LoraConfig, get_peft_model, TaskType

class train_model:
    
     def __initi__(self):
        self.dataset_name = os.getenv("DATASET_NAME")
        self.dataset = self.load_dataset()
        
        self.model_name = os.getenv("MODEL_NAME")
        
        self.original_model , self.tokenizer = self.load_base_model()
        #loading llm
        self.tokenizer , self.peft_model  = self.load_llm()
        self.retriever = self.get_retriver()
        pass

    def load_dataset(self):

        huggingface_dataset_name =  self.dataset_name
        dataset = load_dataset(huggingface_dataset_name)

        return dataset

    def print_number_of_trainable_model_parameters(self, model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


    def data_split(self, dataset):
        dataset_train_test = dataset["train"].train_test_split(test_size=0.2)
        dataset_validation = dataset_train_test["train"].train_test_split(test_size=0.1)
        dataset = DatasetDict({
            'train': dataset_validation['train'],
            'test': dataset_train_test['test'],
            'validation': dataset_validation['test']})

        return dataset

    def load_base_model(self):
        # original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

        original_model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        return original_model , tokenizer


    def ret_data_attribs(self,dataset):

        tokenized_inputs = concatenate_datasets([dataset["train"], dataset["validation"] ,dataset["test"]]).map(lambda x: self.tokenizer(x["question"], truncation=True), batched=True, remove_columns=["question", "answer"])

        max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
        # print(f"Max source length: {max_source_length}")

        tokenized_targets = concatenate_datasets([dataset["train"], dataset["validation"] ,dataset["test"]]).map(lambda x: self.tokenizer(x["answer"], truncation=True), batched=True, remove_columns=["question", "answer"])
        max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

        return max_source_length,max_target_length


    def preprocess_function(sample,padding="max_length"):
        
        max_source_length,max_target_length = self.ret_data_attribs(self.dataset)
        # add prefix to the input for t5
        inputs = ["explain: \n" + item for item in sample["question"]]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["answer"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self,eval_preds):
        metric = evaluate.load("rouge")
        preds, labels = model_preds

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_preds]
        result = metric.compute(predictions = decoded_preds , references=decoded_labels, use_stemmer=True)
        return result
    
    def train_llm(self):
        
        peft_tokenized_datasets = self.dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])
        lora_config = LoraConfig(
            r=64, # Rank
            lora_alpha=64,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
        )

        peft_model = get_peft_model(original_model, 
                                    lora_config)
        
        output_dir = f'./peft-qa-model-training-{str(int(time.time()))}'

        peft_training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3, # Higher learning rate than full fine-tuning.
            num_train_epochs=5,
            logging_steps=1,
            max_steps=10    
        )

        peft_trainer = Seq2SeqTrainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=peft_tokenized_datasets["train"],
            data_collator=data_collator,
            eval_dataset=peft_tokenized_datasets['validation'],
            compute_metrics=compute_metrics
        )

        peft_trainer.train()

        peft_model_path="./peft-qa-model-local2"

        peft_trainer.model.save_pretrained(peft_model_path)
        tokenizer.save_pretrained(peft_model_path)
