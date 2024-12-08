import re
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification


topic_encode = {
    'algebra': 0,
    'polynomial_sequences_calculus': 1,
    'number_theory': 2,
    'geometry': 3,
    'measurement': 4,
    'probability_combinatorics': 5,
    'comparison_reasoning': 6
}

def model_predict(data, tokenizer, model0, model1, return_label = False):
    '''
    description:
        The function will take list of data and output its prediction,
    required:
        remove_formatting() function
    input:
        data: list of text.
        tokenizer: tokenizer from DistilBertTokenizer
        model0: model from TFDistilBertForSequenceClassification
        model1: model from TFDistilBertForSequenceClassification
        return_label: False will return probability, True will return label
    output:
        output: output of model
    '''
    data_cleared = pd.Series(data).apply(remove_formatting)
    data_encoded = tokenizer(list(data_cleared), truncation=True, padding=True,return_tensors="tf")

    #predict
    output_0 = model0(data_encoded)[0]
    output_1 = model1(data_encoded)[0]

    output_all = output_0 + output_1

    if return_label == True:
        output_all = np.argmax(output_all,axis=1)
    return output_all

#function remove all formating character from text
def remove_formatting(text):
    # Remove LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Remove curly braces
    text = re.sub(r'{|}', '', text)
    # Remove Markdown formatting (e.g., **bold**, *italic*, `code`)
    text = re.sub(r'\*\*|\*|`', '', text)
    # Remove Markdown headers (e.g., # Header)
    text = re.sub(r'#', '', text)
    # Remove escape characters
    text = re.sub(r'\\', '', text)
    # remove special character
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\$', '', text)
    # Remove everything after '(A)'
    text = re.sub(r'\(A\).*', '', text)
    text = re.sub(r'\(a\).*', '', text)
    # Remove everything after "b'"
    text = re.sub(r'b\'', '', text)
    return text