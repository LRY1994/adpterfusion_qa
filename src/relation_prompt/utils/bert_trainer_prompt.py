
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from pytorch_metric_learning import miners, losses

# from .abstract_processor import convert_examples_to_features
from .common import timeit
import wandb


class BertTrainer(object):
    def __init__(self, model, optimizer, processor, tokenizer, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.tokenizer = tokenizer
        self.total_step = 0
        self.iterations, self.nb_tr_steps, self.tr_loss = 0, 0, 0
        self.best_dev_f1, self.unimproved_iters = 0, 0
        self.early_stop = False
        self.loss = losses.NTXentLoss(temperature=0.04)

    @timeit
    def train_epoch(self, train_dataloader,epoch,group_idx):
        self.tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.total_step += 1
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, input_mask, label_ids, label_mask = batch
            # print(input_ids)
            inputs = {
                "input_ids": input_ids,
                'mode': 'query'
            }
            labels = {
                "input_ids": label_ids,
                'mode': 'answer'
            }
            
            if self.args.amp:
                with autocast():
                    sequence_output1 = self.model(**inputs).last_hidden_state 
                    sequence_output2 = self.model(**labels).last_hidden_state 
            else:
                sequence_output1 = self.model(**inputs).last_hidden_state
                sequence_output2 = self.model(**labels).last_hidden_state
            query_embed1 = sequence_output1[:, -1]# head entity  + relation prompt + [MASK] [CLS] 
            query_embed2 = sequence_output2[:, -1]# tail entity [CLS]
            query_embed = torch.cat([query_embed1, query_embed2], dim=0)
            # query_embed : [2 * batch_size, hidden]

            label_index = torch.arange(query_embed1.size(0))
            label_index = torch.cat([label_index, label_index], dim=0)
            loss = self.loss(query_embed, label_index)
            
           

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()

            if (self.total_step % self.args.save_step == 0) and (
                self.total_step < 20000
            ):
                self.save_model(step=self.total_step)
            self.tr_loss += loss.item()
            self.nb_tr_steps += 1
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

        wandb.log({f'group-{group_idx}':self.tr_loss},step = epoch )
        print(self.tr_loss)
        
        

    def train_subgraph_cache_tokens(self, group_idx):
        tokenized_features = self.processor.load_and_cache_tokenized_features(
            group_idx, self.tokenizer, self.args
        )
        self.num_train_optimization_steps = (
            int(
                len(tokenized_features)
                / self.args.batch_size
                / self.args.gradient_accumulation_steps
            )
            * self.args.epochs
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_optimization_steps,
        )
        print(f"Start Training on group_idx {group_idx}")
        self.train(tokenized_features, group_idx)
    
    # train_subgraph->train_epoch
    def train_subgraph(self, group_idx):
        '''
        InputExample(
                    guid=None,
                    text_e=text_h + '<mask>',
                    text_r=text_r,
                    label=text_t,
                )
        '''       
        def collate_fn_batch_encoding(batch):
            batch_text = [example.text_e for example in batch]
            # text_h + '<mask>'
            text_features = self.tokenizer.batch_encode_plus(
                batch_text,
                padding="max_length",  # First sentence will have some PADDED tokens to match second sequence length
                max_length=self.args.max_seq_length,
                return_tensors="pt",
                truncation=True,
            )
            #text_t        
            labels = [example.label for example in batch]           
            label_features = self.tokenizer.batch_encode_plus(
                labels,
                padding="max_length",  # First sentence will have some PADDED tokens to match second sequence length
                max_length=self.args.max_seq_length,
                return_tensors="pt",
                truncation=True,
            )
            # for example in batch:
            #     n_cls, idx = example.label
            #     label = np.zeros(n_cls)
            #     label[idx] = 1
            #     labels.append(label)
            # label_ids = torch.as_tensor(labels, dtype=torch.long)
           
            return (
                text_features.input_ids,
                text_features.attention_mask,
                label_features.input_ids,
                label_features.attention_mask,
            )
          
       
        examples = self.processor._create_examples(group_idx)
        self.num_train_optimization_steps = (
            int(
                len(examples)
                / self.args.batch_size
                / self.args.gradient_accumulation_steps
            )
            * self.args.epochs
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_optimization_steps,
        )
        train_dataloader = DataLoader(
            examples,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn_batch_encoding,
        )

        print(f"Start Training on group_idx {group_idx}")
        # path = os.path.join(os.getcwd(), "prompt.txt")
        # with open(path, "a", encoding="utf8", errors="ignore") as writer:
        #     writer.write(f"group_{group_idx}_init")
        #     writer.write(str(self.model.prompt.grad))
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):
            self.train_epoch(train_dataloader,epoch, group_idx )
            self.save_model(epoch=epoch, group_idx=group_idx)
            # with open(path, "a", encoding="utf8", errors="ignore") as writer:
            #         writer.write(f"group_{group_idx}_{self.args.epochs-1}")
            #         writer.write(str(self.model.prompt.grad))
            

    def train(self, tokenized_features, group_idx):
        for epoch in tqdm(range(self.args.epochs), file=sys.stdout, desc="Epoch"):           
            train_sampler = RandomSampler(tokenized_features)
            train_dataloader = DataLoader(
                tokenized_features,
                sampler=train_sampler,
                batch_size=self.args.batch_size,
            )
            self.train_epoch(train_dataloader,epoch,group_idx)
            self.save_model(epoch=epoch, group_idx=group_idx)
          

                          
        


    def save_model(self, epoch=None, step=None, group_idx=None):
        if epoch is not None and group_idx is not None:
            save_path = f"{self.args.save_path}/group_{group_idx}_epoch_{epoch}/"
        elif step is not None:
            save_path = f"{self.args.save_path}/step_{step}/"

        print(f"saving model to {save_path}")
        os.makedirs(save_path, exist_ok=True)


        if self.args.n_gpu > 1:
            self.model.module.save_adapter(save_path, self.args.adapter_names)
        else:
            self.model.save_adapter(save_path, self.args.adapter_names)
