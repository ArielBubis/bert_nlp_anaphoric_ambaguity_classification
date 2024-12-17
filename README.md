# Requirement Analyzer

Welcome to the Requirement Analyzer GUI! This application allows users to upload Excel or CSV files containing requirements and analyze them using Natural Language Processing (NLP) and Machine Learning (ML) models, specifically a BERT-based model. The tool identifies ambiguous requirements for further processing, and uses NLP tools to find possible candidates resolve anaphora.

## Table of Contents

- [Features](#features)
- [Model Details](#model-details)
- [Datasets](#datasets)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
  
## Features

**User-Friendly GUI**: Easy-to-use interface built with Tkinter for uploading files and running models.
**Ambiguity Detection**: Employs a custom BERT-based model to identifies and processes ambiguous requirements separately.
**Automated NLP Analysis**: Uses SpaCy to process text and find possible candidates resolve anaphora.
**CSV Output**: Saves processed results in CSV format for further analysis.

## Model Details
[link for the colab notebook used for the model finetuning](https://colab.research.google.com/drive/1gaO63M4Sh-k_Pk_lYcP8ygPb0NxqO-Pt?usp=sharing&authuser=1#scrollTo=RpLAUIrPvqDc)

This project uses a custom BERT-based model implemented in PyTorch for intent classification. The model has been fine-tuned on a dataset of requirements to classify them as either "ambiguous" or "unambiguous."
Model Architecture:

  - Pre-trained Model: DistilBERT (distilbert-base-uncased)
  - Custom Layers: Fully connected layers with ReLU activation and dropout for regularization.
  - Output Layer: Softmax activation for binary classification.
## Datasets
* DAMIR (Dataset for Anaphoric aMbiguity In Requirements)
* ReqEval - RE dataset on anaphoric ambiguity released in the 2020 edition of the NLP4RE workshop.

# Model Performance
## Evaluation
We evaluated both ChatGPT and our model's performance using the same 20 requirements from the "ReqEval" dataset, applying a zero-shot classification approach to compare their ability to identify ambiguous and unambiguous requirements.
### ChatGPT Performance
![Untitled](https://github.com/user-attachments/assets/cf267b88-b953-400f-bc6c-046539b31956)
* Precision: 0.8 (High accuracy with clear requirements)
* Recall: 0.4 (Struggles with ambiguous requirements)
* Tends to "force" understanding of unclear specifications
### Our Model Performance
* Accuracy: 0.737
* Recall: 0.737
* Precision: 0.737
![Untitled](https://github.com/user-attachments/assets/c53d305b-df5a-4f67-bcc7-9f2aa978ea1c)

## Key Insights
Our model provides a more balanced approach to identifying ambiguous requirements. Unlike ChatGPT, which misses many ambiguous cases, we prioritize recall to catch potential issues early in the project lifecycle.
The consistent 0.737 score across metrics indicates a reliable method for detecting requirement ambiguities, helping prevent costly corrections in later development stages.


# Installation

### Option 1: EXE Version (Recommended for Easy Setup)

**Download an EXE version with everything built in for a one-click setup.** 

- Note that it may take some time to load up the GUI, so please be patient when using it.
- the file is also fairy large (1.75 GB) as it hold all the libraries and the model built in.

[Download EXE Version Here](https://drive.google.com/file/d/12jaryZOWv80JuvGE1GdFEkZL1OOnkuS-/view?usp=sharing)

### Option 2: Manual Setup

If you prefer to set up the project manually or customize the environment, follow these steps:

**Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/bert_nlp_anaphoric_ambaguity_classification.git
   cd bert_nlp_anaphoric_ambaguity_classification
   ```

Step 1: Set Up the Virtual Environment

On the first run, install Python 3.12 (will fail with python 3.13) set up the virtual environment and install the required dependencies by running the following command:
 
   ```bash
   python3.12 setup_venv.py
   ```

### What This Script Will Do:

1. Check if a virtual environment is already activated.
2. If not, it will create a new virtual environment in the current directory.

### Step 2: Set the Interpreter to the Virtual Environment

Ensure that your Python interpreter is set to the newly created virtual environment.

### Step 3: Download the Model and Label Encoder

Download the model checkpoint and label encoder and place them in the /model folder. 

- model:
https://drive.google.com/file/d/17UDOKfjVnNorOYwPHeAZKcfuTm_oHpE1/view?usp=sharing

- label encoder:
https://drive.google.com/file/d/1gxJNMHJannNJzTdJD50nB2oH7RMEDspe/view?usp=sharing

Download [SpaCy](https://pypi.org/project/spacy/) + en_core_web_sm for the nlp model using the following commands
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Step 4: Run the Application
After setting up the virtual environment, 
you can run the GUI application:
  ```bash
  python GUI_v3.py
  ```
## Explanation

### setup_venv.py
This script checks if a virtual environment is activated. If not, it creates a new virtual environment in the current directory

### GUI_v3.py
This script provides a GUI for the Requirement Analyzer application. It includes the following functionalities:

  - NLP Model Initialization: Loads the en_core_web_sm model from SpaCy, downloading it if necessary.
  - File Upload: Allows users to upload an Excel or CSV file containing requirements.
  - Model Execution: Processes the uploaded file using a BERT model to identify intents and resolve ambiguous intents.
  - NLP Processing: Uses SpaCy to resolve anaphora in the requirements and saves the results to CSV files.  

    
### BERT_Arch.py
This script defines a custom BERT architecture for the requirement analysis model. It includes:

- A dropout layer for regularization.
- ReLU activation functions.
- Dense layers for transforming the BERT embeddings.
- A softmax activation function for the output layer.
  
### Additional Features

  - NLP Model Initialization: Automatically initializes and downloads the required NLP model if not already installed.
  - File Upload: Supports uploading Excel and CSV files.
  - Requirement Analysis: Processes the requirements and identifies intents using a pre-trained model.
  - Ambiguous Intent Resolution: Resolves ambiguous intents using NLP techniques and saves the results to CSV files.

## Usage
![WhatsApp Image 2024-08-27 at 13 04 48](https://github.com/user-attachments/assets/1b2c9ab8-c649-4634-9786-20b18f114d93)
1. Upload File: Click the "Upload File" button to upload an Excel or CSV file containing the requirements.
2. Run Model: Click the "Run Model" button to process the uploaded file and analyze the requirements.
### Features
  - NLP Model Initialization: Automatically initializes and downloads the required NLP model if not already installed.
  - File Upload: Supports uploading Excel and CSV files.
  - Requirement Analysis: Processes the requirements and identifies intents using a pre-trained model.
  - Ambiguous Intent Resolution: Resolves ambiguous intents using NLP techniques and saves the results to CSV files.


## References
Ezzini, S., Abualhaija, S., Arora, C., & Sabetzadeh, M. (2022). TAPHSIR: Towards AnaPHoric Ambiguity Detection and ReSolution In Requirements. arXiv:2206.10227v1 [cs.SE]. Available at: https://arxiv.org/abs/2206.10227
Ezzini, S., Abualhaija, S., Arora, C., & Sabetzadeh, M. (2022). Automated Handling of Anaphoric Ambiguity in Requirements: A Multi-solution Study. IEEE/ACM 44th International Conference on Software Engineering (ICSE). Available at: https://dl.acm.org/doi/10.1145/3510003.3510157
Qiu, K., Puccinelli, N., Ciniselli, M., & Di Grazia, L. (2024). From Today's Code to Tomorrow's Symphony: The AI Transformation of Developer's Routine by 2030. arXiv:2405.12731v2 [cs.SE]. Available at: https://arxiv.org/pdf/2405.12731
Sridhara, G., H.G., R., & Mazumdar, S. (2023). ChatGPT: A Study on its Utility for Ubiquitous Software Engineering Tasks. arXiv:2305.16837v1 [cs.SE]. Available at: https://arxiv.org/pdf/2305.16837

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
