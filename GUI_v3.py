print("Welcome to the Requirement Analyzer GUI!")
print("Please wait while the models are loaded")
print("This may take a few seconds...")


import os
import sys
import datetime
import subprocess

# Now import the necessary libraries
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import spacy
from model_utils.useModel import get_prediction  # Assuming this function is in useModel.py

# Install required packages
def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All dependencies are installed.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)


print("Loading BERT model...")

def initialize_en_core_model():
    print("Loading NLP model...")
    global nlp
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'model_utils', 'en_core_web_sm', 'en_core_web_sm-3.8.0')
        print(f"The 'en_core_web_sm' model is located at: {model_path}")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Error loading the 'en_core_web_sm' model from {model_path}: {e}")
        sys.exit(1)  # Exit the program if the model cannot be loaded

# Process uploaded file and run BERT model
def run_model(filepath):
    try:
        status_var.set("Processing...")
        df = pd.read_excel(filepath)
        df = df.drop_duplicates()  # Remove duplicates
        sentences = df['Requirements']  # Assuming requirements are in the 'Requirements' column
        results = {'Sentence': [], 'Intent': []}
        for sentence in sentences:
            intent = get_prediction(sentence)
            results['Sentence'].append(sentence)
            results['Intent'].append(intent)
        results_df = pd.DataFrame(results)
        results_df['Intent'] = results_df['Intent'].replace({0: 'ambiguous', 1: 'unambiguous'})
        ambiguous_df = results_df[results_df['Intent'] == 'ambiguous']
        
        # Save to CSV
        # Ensure the processed_data and output directories exist
        processed_data_dir = 'processed_data'
        output_dir = 'output'
        os.makedirs(processed_data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the initial processed file in the processed_data directory
        results_df.to_csv(os.path.join(processed_data_dir, f'all_intents_{timestamp}.csv'), index=False)
        ambiguous_df.to_csv(os.path.join(processed_data_dir, f'ambiguous_intents_{timestamp}.csv'), index=False)
        status_var.set("Model run complete. Proceeding to NLP processing...")

        # Process ambiguous intents and save the resolved anaphora file in the output directory
        process_ambiguous_intents(
            os.path.join(processed_data_dir, f'ambiguous_intents_{timestamp}.csv'),
            os.path.join(output_dir, f'resolved_anaphora_{timestamp}.csv')
        )
    except Exception as e:
        status_var.set(f"Error during model run: {str(e)}")

# Find pronouns and resolve anaphora using NLP
def findPronouns(sent, pronouns):
    tokens = []
    for t in sent:
        if "PRP" in t.tag_ and t.text.lower() in pronouns and t not in tokens:
            tokens.append(t)
    return tokens

def applynlp(string, nlp):
    tr = np.nan
    try:
        tr = nlp(string)
    except:
        print(string)
    return tr

def getNPs(sent, p, include_nouns=False):
    nps = []
    npstr = []
    chunks = list(sent.noun_chunks)
    for i in range(len(chunks)):
        np = chunks[i]
        if np.end <= p.i:
            if len(np) == 1:
                if np[0].pos_ not in ["NOUN", "PROPN"]:
                    continue
            if np.text.lower() in npstr:
                for x in nps:
                    if x.text.lower() == np.text.lower():
                        nps.remove(x)
                npstr.remove(np.text.lower())
            nps.append(np)
            npstr.append(np.text.lower())
            if i < len(chunks) - 1:
                np1 = chunks[i + 1]
                if np1.start - np.end == 1:
                    if sent.doc[np.end].tag_ == "CC":
                        newnp = sent.doc[np.start:np1.end]
                        if newnp.text.lower() in npstr:
                            for x in nps:
                                if x.text.lower() == newnp.text.lower():
                                    nps.remove(x)
                            npstr.remove(newnp.text.lower())
                        nps.append(newnp)
                        npstr.append(newnp.text.lower())
    if include_nouns:
        for t in sent:
            if t.i < p.i and "subj" in t.dep_ and t.pos_ == "NOUN":
                if t.text.lower() in npstr:
                    for x in nps:
                        if x.text.lower() == t.text.lower():
                            nps.remove(x)
                    npstr.remove(t.text.lower())
                npstr.append(t.text.lower())
                nps.append(sent[t.i:t.i + 1])
    return nps

def create_csv(exampleData, pronouns, output_path):
    li = []
    i, j = 0, 0
    ids = []
    for context in exampleData.Requirement.unique():
        for pronoun in findPronouns(context, pronouns):
            Id = str(i) + "-" + pronoun.text + "-" + str(j)
            while Id in ids:
                j += 1
                Id = str(i) + "-" + pronoun.text + "-" + str(j)
            for candidateAntecedent in getNPs(context, pronoun):
                li.append([Id, context, pronoun, pronoun.i, candidateAntecedent])
                ids.append(Id)
        i += 1
    result_df = pd.DataFrame(li, columns=["Id", "Context", "Pronoun", "Position", "Candidate Antecedent"])
    result_df.to_csv(output_path, index=False)

def process_ambiguous_intents(input_path, output_path):
    try:
        exampleData = pd.read_csv(input_path)
        exampleData["Requirement"] = exampleData["Sentence"].apply(lambda x: applynlp(x, nlp))
        create_csv(exampleData, pronouns, output_path)
        status_var.set("Upload an Excel/CSV file and run the model.")
        messagebox.showinfo("Success", f"Processing complete. Results saved to {output_path}")
    except Exception as e:
        status_var.set(f"Error during NLP processing: {str(e)}")

# GUI Functions
def upload_file():
    global filepath
    filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv")])
    if filepath:
        status_var.set("File uploaded successfully. Click 'Run Model' to start processing.")

def run_pipeline():
    if not filepath:
        messagebox.showwarning("Input Error", "Please upload an Excel/CSV file first")
        return
    
    run_model(filepath)

def main():
    global status_var, pronouns
    install_requirements()

    root = tk.Tk()
    root.title("Requirement Analyzer")

    status_var = tk.StringVar()

    upload_button = tk.Button(root, text="Upload File", command=upload_file)
    upload_button.pack()

    run_button = tk.Button(root, text="Run Model", command=run_pipeline)
    run_button.pack()

    status_label = tk.Label(root, textvariable=status_var)
    status_label.pack()

    # Initialize en_core model
    initialize_en_core_model()

    print("NLP model loaded successfully.")

    # Define pronouns
    pronouns = ["I", "me", "my", "mine", "myself", "you", "you", "your", "yours", "yourself", 
                "he", "him", "his", "his", "himself", "she", "her", "her", "hers", "herself", 
                "it", "it", "its", "itself", "we", "us", "our", "ours", "ourselves", "you", 
                "you", "your", "yours", "yourselves", "they", "them", "their", "theirs", "themselves"]

    # Setup GUI
    status_var.set("Upload an Excel/CSV file and run the model.")

    root.mainloop()
if __name__ == "__main__":
    main()