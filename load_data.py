import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
BOS = '<extra_id_0>'

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data_dir = data_folder
        self.split = split
        self.data = self.process_data(data_folder, split, self.tokenizer)
        

    def process_data(self, data_folder, split, tokenizer):
        data_in = []
        data_out = []
        data = []
        
        if split == 'test':
            data_in = load_lines(os.path.join(data_folder, 'test.nl'))
        else:
            data_in = load_lines(os.path.join(data_folder, f'{split}.nl'))
            data_out = load_lines(os.path.join(data_folder, f'{split}.sql'))
            
        
        if split=='test':
            
            for nl_in in data_in:
                encoder_in = tokenizer(f"Convert the following English to SQL query: {nl_in}",truncation=True, max_length=512, return_tensors='pt')
                decoder_begin = tokenizer.convert_tokens_to_ids([BOS])[0]
                data.append({
                    'encoder_input_ids': encoder_in['input_ids'].squeeze(0),
                    'encoder_attention_mask': encoder_in['attention_mask'].squeeze(0),
                     'decoder_start_token': decoder_begin
                }) # type: ignore
        else:
            
            for nl_in, sql_out in zip(data_in, data_out):
                encoder_in = tokenizer(f"Convert the following English to SQL query: {nl_in}", truncation=True, max_length=512 , return_tensors='pt')
                decoder_out = tokenizer(sql_out, return_tensors='pt')
                decoder_begin = tokenizer.convert_tokens_to_ids([BOS])[0]
                decoder_input_ids = torch.cat([torch.tensor([decoder_begin]), decoder_out['input_ids'].squeeze(0)])
                decoder_target_ids = torch.cat([decoder_out['input_ids'].squeeze(0), torch.tensor([tokenizer.eos_token_id])])
                
                data.append({
                    'encoder_input_ids': encoder_in['input_ids'].squeeze(0),
                    'encoder_attention_mask': encoder_in['attention_mask'].squeeze(0),
                    'decoder_input_ids': decoder_input_ids,
                    'decoder_target_ids': decoder_target_ids,
                     'decoder_start_token': decoder_begin
                })
        
        return data 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_masks = [item['encoder_attention_mask'] for item in batch]
    decoder_inputs = [item['decoder_input_ids'] for item in batch]
    decoder_targets = [item['decoder_target_ids'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    
    initial_decoder_inputs = torch.tensor([item['decoder_start_token'] for item in batch])
    
    return encoder_ids, encoder_masks, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_input_ids'] for item in batch]
    encoder_masks = [item['encoder_attention_mask'] for item in batch]
    
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    
    initial_decoder_inputs = torch.tensor([item['decoder_start_token'] for item in batch])
    
    return encoder_ids, encoder_masks, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
                         
    return train_x, train_y, dev_x, dev_y, test_x