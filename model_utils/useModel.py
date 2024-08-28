import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.preprocessing import LabelEncoder
import pickle
from .BERT_Arch import BERT_Arch

# Determine if the script is running in a PyInstaller bundle
if getattr(sys, 'frozen', False):
    # If the script is running in a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # If the script is running in a normal Python environment
    base_path = os.path.abspath(os.path.dirname(__file__))

# Construct paths to the model and label encoder
label_encoder = LabelEncoder()
label_encoder_path = os.path.join(base_path, '../model', 'label_encoder(3).pkl')
model_path = os.path.join(base_path, '../model', 'best_model_checkpoint_distilbert-base-uncased_lr_0.0008423123747876546_bs_16_2024-08-28_12-18-36.pth')
# # Construct paths to the model and label encoder only for the exe file
# label_encoder_path = os.path.join(base_path, 'model', 'label_encoder(3).pkl')
# model_path = os.path.join(base_path, 'model', 'best_model_checkpoint_distilbert-base-uncased_lr_0.0008423123747876546_bs_16_2024-08-28_12-18-36.pth')

label_encoder.fit(['ambiguous', 'unambiguous'])
# Load the label encoder
# print(f"Loading label encoder from: {label_encoder_path}")
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
# Specify GPU or CPU
device = torch.device("cpu")

# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the BERT tokenizer

# # Load the BERT tokenizer and model
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# bert = AutoModel.from_pretrained('bert-base-uncased')

# Initialize the custom BERT architecture
model = BERT_Arch(bert)

# Push the model to CPU (or GPU if available)
model = model.to(device)

# Load the trained model's state_dict
# print(f"Loading model from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

max_seq_len = 60

def get_prediction(input_text):
    model.eval()
    if len(input_text) <= 20:
        return "Error: Input text must be longer than 20 characters."
    
    # Tokenize the input text
    tokens_test_data = tokenizer(
        [input_text],
        max_length=max_seq_len,
        padding='max_length', 
        truncation=True,
        return_token_type_ids=False
    )
    
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])

    # Make predictions
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
    
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    intent = label_encoder.inverse_transform(preds)[0]
    
    return intent
