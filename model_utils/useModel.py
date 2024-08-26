# # Install Transformers
# !pip install transformers
# # To get model summary
# !pip install torchinfo
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
from sklearn.preprocessing import LabelEncoder
import pickle
# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Print Python path to check if the directory is included
print("Python Path:", sys.path)

# Adding the current working directory to Python path
sys.path.insert(0, os.getcwd())
from model_utils.BERT_Arch import BERT_Arch




# Now try importing again
from model_utils.BERT_Arch import BERT_Arch
# specify GPU or CPU
device = torch.device("cpu")

# # Import the DistilBert pretrained model
# bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# # Load the DistilBert tokenizer
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# Import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

from sklearn.preprocessing import LabelEncoder

with open('model/label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)
   
for param in bert.parameters():
    param.requires_grad = False
model = BERT_Arch(bert)
# push the model to GPU
model = model.to(device)
from torchinfo import summary
# summary(model)

model.load_state_dict(torch.load("model/model_2_7_24_13_00.pth",map_location=torch.device('cpu')))

max_seq_len = 60

def get_prediction(str):
    # str = re.sub(r'[^a-zA-Z ]+', '', str)
    # print (str)
    test_text = [str]
    model.eval()
    if len(str) <= 20:
        intent =  "Error: Input text must be longer than 20 characters."
    else:
        tokens_test_data = tokenizer(
            test_text,
            max_length = max_seq_len,
            padding='max_length', 
            truncation=True,
            return_token_type_ids=False
        )
        test_seq = torch.tensor(tokens_test_data['input_ids'])
        test_mask = torch.tensor(tokens_test_data['attention_mask'])

        preds = None
        with torch.no_grad():
            preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        intent = le.inverse_transform(preds)[0]
    # print("Intent Identified: ", intent)
    return intent

