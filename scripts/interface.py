
import time

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
#import tensorflow as tf
from transformers import GPT2Tokenizer
import time
import random
import matplotlib.pyplot as plt
import torch
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2LMHeadModel


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

import os
def load_models_only(seed_val):
    ## Model and tokeniser
    #tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    #tokenizer.add_special_tokens(
    #	{'additional_special_tokens': ['<|startofing|>', '<|endofing|>', '<|ingseparator|>']}
    #)

    tokenizer = GPT2Tokenizer.from_pretrained("./recipe_generator/gpt2_models/tokenizer")
    # instantiate the model
    #configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    #model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)#"./recipe_generator/gpt2_models/TrainRecipeBox/Ing")#"gpt2", config=configuration)#('./models/RecipeBoxTrained')#
    model = GPT2LMHeadModel.from_pretrained("./recipe_generator/gpt2_models/model")

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    # Tell pytorch to run this model on the GPU.
    device = torch.device("cpu")
    #model.cuda()

    # Set the seed value all over the place to make this reproducible.
    #seed_val = 45git status

    #random.seed(seed_val)
    #np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    return model, tokenizer, device


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
        top_k=50, max_length = 2000, top_p=0.95, num_return_sequences=1
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
    seed_val = random.randint(1, 100)
    print(seed_val)
    print("seed_val: ", seed_val)
    model, tokenizer, device = load_models_only(seed_val)
    ing_list = ing_list.lower().split(', ')
    return generate_sample_from_user(model, ing_list, tokenizer, device, complete=complete)

    # time.sleep(5)
    # return "<|startofing|> ing1 <|ingseparator|> ing2 <|ingseparator|> ing3 <|endofing|> <|startoftext|> step1. step2. step3 <|endoftext|>" #



if __name__ == '__main__':
    #model, validation_dataloader, tokenizer, device = load_data_and_models()
    #generate_sample_from_val(model, validation_dataloader, tokenizer, device)

    model, tokenizer, device = load_models_only()

    ing_list = input("Please list ingredients, lowercase, separated by commas")
    ing_list = ing_list.split(', ')
    
    complete_q = input("Is this all the ingredients? y/n")

    complete = True if complete_q.lower() == 'y' else False

    print(generate_sample_from_user(model, ing_list, tokenizer, device, complete=complete))



