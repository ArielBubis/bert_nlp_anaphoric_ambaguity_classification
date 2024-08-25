import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import spacy
import os


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' is not installed. Attempting to download.")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Load the model from the directory
nlp = spacy.load("en_core_web_sm")
pronouns=["I","me","my","mine","myself","you","you","your","yours","yourself","he","him","his","his","himself","she","her","her","hers","herself","it","it","its","itself","we","us","our","ours","ourselves","you","you","your","yours","yourselves","they","them","their","theirs","themselves"]

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


def run_script(input_path, output_path):
    exampleData = pd.read_csv(input_path, names=["Requirement"], sep="\t")
    exampleData["Requirement"] = exampleData["Requirement"].apply(lambda x: applynlp(x, nlp))
    create_csv(exampleData, pronouns, output_path)
    messagebox.showinfo("Success", f"File processed and saved to {output_path}")

def browse_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def browse_save_file(entry):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

def run():
    input_path = input_entry.get()
    output_path = output_entry.get()
    if not input_path or not output_path:
        messagebox.showwarning("Input Error", "Please select both input and output files")
        return
    run_script(input_path, output_path)

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    root.title("CSV Processor")

    tk.Label(root, text="Input CSV File:").grid(row=0, column=0, padx=10, pady=5)
    input_entry = tk.Entry(root, width=50)
    input_entry.grid(row=0, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: browse_file(input_entry)).grid(row=0, column=2, padx=10, pady=5)

    tk.Label(root, text="Output CSV File:").grid(row=1, column=0, padx=10, pady=5)
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=1, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=lambda: browse_save_file(output_entry)).grid(row=1, column=2, padx=10, pady=5)

    tk.Button(root, text="Run", command=run).grid(row=2, column=1, pady=10)

    # Run the Tkinter event loop
    root.mainloop()