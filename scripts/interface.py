import numpy as np
import pandas as pd
import ast
import _pickle as pickle
from os import path
import time

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer

import copy
import operator
import datetime
import warnings
from itertools import product
warnings.filterwarnings("ignore")
#import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import os
import time
import datetime
import random
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    #self.label_ids = []
    self.attn_masks = []

    for index, item in txt_list.iterrows():
      encodings_dict = tokenizer('<|startofing|>'+ '<|ingseparator|>'.join(item['ingredients']) + '<|endofing|>' + '<|startoftext|>'+ item['recipe_steps'] + '<|endoftext|>', truncation=True, max_length=850, padding="max_length")
      #print('<|startofing|>'+ ' <|ingseparator|> '.join(item['ingredients']) + '<|endofing|>')
      #print('<|startoftext|>'+ item['recipe_steps'] + '<|endoftext|>')
      #encodings_labels = tokenizer('<|startoftext|>'+ item['recipe_steps'] + '<|endoftext|>', truncation=True, max_length=750, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      #self.label_ids.append(torch.tensor(encodings_labels['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 

def load_recipes():
    return None

def clean_ing(ing):
    ing_list = ing.split()
    clean = ' '.join(list(filter(lambda x: x.lower() != "advertisement", ing_list)))
    return clean

def clean_ing_list(ing_list):
    ing_list = ing_list.split('; ')
    ing_list.remove('')
    ing_list = [clean_ing(ing) for ing in ing_list if ing != 'advertisement']
    return ing_list

def load_data_and_models():
    train_data = load_recipes()
    ing_data = [clean_ing_list(ing_list) for ing_list in train_data[1]]
    numWords = np.array([len(sentence.split()) for sentence in train_data[2]])

    recipe_steps = pd.DataFrame()
    recipe_steps['ingredients'] = ing_data
    recipe_steps['recipe_steps'] = train_data[2]

    indexes =np.where(numWords>750)[0]
    recipe_steps = recipe_steps.drop(indexes)



    ## Model and tokeniser
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    #tokenizer.add_special_tokens(
    #	{'additional_special_tokens': ['<|startofing|>', '<|endofing|>', '<|ingseparator|>']}
    #)
    tokenizer = GPT2Tokenizer.from_pretrained("./recipe_generator/gpt2_models/tokenizer/Ing")
    dataset = GPT2Dataset(recipe_steps, tokenizer, max_length=850)

    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    batch_size = 2

    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    # I'm not really doing anything with the config buheret
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("./recipe_generator/gpt2_models/TrainRecipeBox/Ing")#"gpt2", config=configuration)#('./models/RecipeBoxTrained')#

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cpu")
    #model.cuda()

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    return model, validation_dataloader, tokenizer, device

import os
def load_models_only(seed_val):
    ## Model and tokeniser
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    #tokenizer.add_special_tokens(
    #	{'additional_special_tokens': ['<|startofing|>', '<|endofing|>', '<|ingseparator|>']}
    #)

    tokenizer = GPT2Tokenizer.from_pretrained("./recipe_generator/gpt2_models/tokenizer/Ing")
    # instantiate the model
    #configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    #model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)#"./recipe_generator/gpt2_models/TrainRecipeBox/Ing")#"gpt2", config=configuration)#('./models/RecipeBoxTrained')#
    model = GPT2LMHeadModel.from_pretrained("./recipe_generator/gpt2_models/TrainRecipeBox/Ing")

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cpu")
    #model.cuda()

    # Set the seed value all over the place to make this reproducible.
    #seed_val = 45

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    return model, tokenizer, device

## Generateion

def generate_sample_from_val(model, validation_dataloader, tokenizer, device):
    itera = iter(validation_dataloader)
    for i in range(np.random.randint(1, len(validation_dataloader)-1)):
        next(itera)

    example = next(itera)[0][0]
    ing_end_tok = tokenizer.encode('<|endofing|>')[0]
    pad_tok = tokenizer.encode('<|pad|>')

    b = example == ing_end_tok
    loc = b.nonzero()[0] + 2

    input_ex = tokenizer.decode(example[:loc])
    input_ex = torch.LongTensor(tokenizer.encode(input_ex))#, padding="max_length", max_length=850))
    input_ex = torch.reshape(input_ex, [1, len(input_ex)])
    input_ex = torch.cat([input_ex, input_ex], 0)

    model = model.to(device)
    s_input_ids = input_ex.to(device)

    t0 = time.time()

    model.eval()

    sample_outputs = model.generate(
        s_input_ids,#bos_token_id=random.randint(1,30000),
        do_sample=True,   
        top_k=50, max_length = 200, top_p=0.95, num_return_sequences=1
                        )

    for i, sample_output in enumerate(sample_outputs):
        print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=False)))


def generate_sample_from_user(model, ing_list, tokenizer, device, complete=True):
    
    ing_list_str = '<|ingseparator|>'.join(ing_list)

    if complete:
        input_str = '<|startofing|>' + ing_list_str + '<|endofing|> <|startoftext|>'
    else:
        input_str = '<|startofing|>' + ing_list_str + '<|ingseparator|>'

    
    input_ex = torch.LongTensor(tokenizer.encode(input_str))
    input_ex = torch.reshape(input_ex, [1, len(input_ex)])
    input_ex = torch.cat([input_ex, input_ex], 0)

    model = model.to(device)
    s_input_ids = input_ex.to(device)

    t0 = time.time()

    model.eval()

    sample_outputs = model.generate(
        s_input_ids,#bos_token_id=random.randint(1,30000),
        do_sample=True,   
        top_k=50, max_length = 200, top_p=0.95, num_return_sequences=1
                        )

    output = None
    for i, sample_output in enumerate(sample_outputs):
        if i==1:
            output = tokenizer.decode(sample_output, skip_special_tokens=False)
            output = output.replace('<|pad|>', "")

            #output = output.replace('<|startofing|>', "Ingredients\n  ~")
            #output = output.replace('<|endofing|> <|startoftext|>', "\nRecipe Steps\n   -")
            #output = output.replace('<|endoftext|>', "\n")
            #output = output.replace('<|pad|>', "")
            #output = output.replace(' <|ingseparator|> ', ",\n  ~")

            #output = output.replace('. ', ".\n   -")

    return output
    #print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=False)))

def get_recipe_with_string_input(ing_list, complete):
    #seed_val = random.randint(1, 100)
    #model, tokenizer, device = load_models_only(seed_val)
    #ing_list = ing_list.lower().split(', ')
    time.sleep(5)
    return "<|startofing|> ing1 <|ingseparator|> ing2 <|ingseparator|> ing3 <|endofing|> <|startoftext|> step1. step2. step3 <|endoftext|>" #
#return generate_sample_from_user(model, ing_list, tokenizer, device, complete=complete)


if __name__ == '__main__':
    #model, validation_dataloader, tokenizer, device = load_data_and_models()
    #generate_sample_from_val(model, validation_dataloader, tokenizer, device)

    model, tokenizer, device = load_models_only()

    ing_list = input("Please list ingredients, lowercase, separated by commas")
    ing_list = ing_list.split(', ')
    
    complete_q = input("Is this all the ingredients? y/n")

    complete = True if complete_q.lower() == 'y' else False

    print(generate_sample_from_user(model, ing_list, tokenizer, device, complete=complete))



