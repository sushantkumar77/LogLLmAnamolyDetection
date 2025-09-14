
import streamlit as st
import torch
import pandas as pd
import numpy as np
import re  # For regular expressions to parse log files
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer #logs into vectors
import torch.nn as nn
from transformers import BertForSequenceClassification  # The pre-trained BERT model structure
import os
import json


# Set up the device to use a GPU (cuda) if available, otherwise default to CPU.
# Using a GPU significantly speeds up model inference.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# 3. MODEL CLASS DEFINITION
# ==============================================================================
# A wrapper class to handle the BERT model for log classification.
class LogClassifier:

    def __init__(self, model_name, num_labels=2):
        """
        Initializes the classifier by loading the pre-trained BERT model structure.
        - num_labels: The number of output classes (2 for ANOMALY/NORMAL).
        """
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(device)  # Move the model to the selected device (GPU/CPU)


    """Saves the model's learned weights (state dictionary) to a file."""
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")



    def load(self, path):
        """
        Loads the learned weights from a .pt file into the model structure.
        - map_location=device: Ensures the model loads correctly on either CPU or GPU.
        """
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.to(device)
        print(f"Model loaded from {path}")


# 4. HELP FUNCTIONS

def parse_bgl_logs(file_content):
    """
    Parses the raw text content of a BGL log file using a regular expression.
    - file_content: The string content of the uploaded .log file.
    - Returns: A list of dictionaries, where each dictionary represents a structured log entry.
    """
    logs_data = [] #creating the logs data dictionary for storing the processed logs
    # This regex pattern is designed to capture the specific columns of the BGL log format.
    pattern = re.compile(
        r"[-\w]*\s*(\d+)\s+(\d{4}\.\d{2}\.\d{2})\s+([\w\-\:]+)\s+(\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)\s+([\w\-\:]+)\s+(\w+)\s+(\w+)\s+(\w+)\s+(.*)"
    )


    for line in file_content.split('\n'):
        line = line.strip()
        match = pattern.match(line)
        if match:
            epoch, date, location, timestamp, component, category, source, severity, message = match.groups()

            #appending into logs data dictionary
            logs_data.append({
                'epoch': epoch, 'date': date, 'location': location, 'timestamp': timestamp,
                'component': component, 'category': category, 'source': source,
                'severity': severity, 'message': message
            })
    return logs_data



def predict_anomaly(log_entry, classifier, tokenizer):
    """

    Performs anomaly detection on a single log entry.
    - log_entry: A dictionary containing the parsed log message.
    - classifier: holds the model to predict
    - tokenizer: The BERT tokenizer instance.
    - Returns: A dictionary with the prediction, anomaly status, and probability.
    """
    # 1. Tokenize the log message and do padding if reqd
    message = log_entry['message']
    encoding = tokenizer(
        message,
        truncation=True,  # Truncate messages longer than max_length
        padding='max_length',  # Pad shorter messages to max_length
        max_length=128,
        return_tensors='pt'  # Return PyTorch tensors

    )

    # 2.A model running on the GPU's memory cannot access data stored in the CPU's main memory (RAM)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 3. Get prediction from the model
    # torch.no_grad() disables gradient calculation for faster inference.
    with torch.no_grad():
        outputs = classifier.model(input_ids=input_ids, attention_mask=attention_mask)
        # The prediction is the index of the highest logit value (0 for NORMAL, 1 for ANOMALY)
        _, prediction = torch.max(outputs.logits, dim=1)

    # 4. Get the probability of the prediction
    # Softmax converts raw logits into a probability distribution.
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    # We are interested in the probability of the "ANOMALY" class (index 1)
    anomaly_prob = probs[0][1].item()

    return {
        'is_anomaly': prediction.item(),
        'anomaly_probability': anomaly_prob,
        'prediction': 'ANOMALY' if prediction.item() == 1 else 'NORMAL'
    }


def save_uploaded_file(uploaded_file, save_dir='temp'):
    """
    Saves a file uploaded via Streamlit to a temporary directory on the server.
    This is needed because many libraries require a file path to read from.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def load_hyperparams(file_path):
    """Loads hyperparameters from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


# ==============================================================================
# 5. STREAMLIT APPLICATION MAIN FUNCTION
# ==============================================================================
def main():
    # --- Page Configuration ---
    st.set_page_config(page_title="Log Anomaly Detector", layout="wide")

    # --- Header ---
    st.title("Log Anomaly Detection Dashboard")
    st.write("Upload your trained TinyBERT model, hyperparameters, and log file for analysis")

    # --- File Uploaders ---
    # Create a three-column layout for a clean UI
    col1, col2, col3 = st.columns(3)
    with col1:
        model_file = st.file_uploader("Upload Model (.pt file)", type=['pt'])
    with col2:
        hyperparams_file = st.file_uploader("Upload Hyperparameters (.json)", type=['json'])
    with col3:
        log_file = st.file_uploader("Upload Log File (.log)", type=['log', 'txt'])

    # --- Session State Initialization ---
    # Session state acts as the app's memory, preventing it from losing data
    # or re-running heavy computations (like loading the model) on every interaction.
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'logs_df' not in st.session_state:
        st.session_state.logs_df = None

    # --- Model Loading Logic ---

    #if not in session state

    # This block runs ONLY when all necessary files are uploaded AND the model hasn't been loaded yet.
    if model_file and hyperparams_file and not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            model_path = save_uploaded_file(model_file)  #to the above function save_upload_file we are sending the model_file for processing and copying it to model_path
            hyperparams_path = save_uploaded_file(hyperparams_file)#to the above function save_upload_file we are sending the hyperparam_file for processing
            hyperparams = load_hyperparams(hyperparams_path)#process the hyperparam_path

            # Initialize the tokenizer for the specific TinyBERT model
            tokenizer = BertTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')


            # Initialize the classifier to clasify anamoly or normal and load the trained weights
            classifier = LogClassifier('huawei-noah/TinyBERT_General_4L_312D')
            classifier.load(model_path)#method to load the weights from the .pt file.
            classifier.model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

            # Store the loaded objects in the session state for later use
            st.session_state.classifier = classifier
            st.session_state.tokenizer = tokenizer
            st.session_state.hyperparams = hyperparams
            st.session_state.model_loaded = True

            st.success("Model loaded successfully!")

    # --- Log Processing and Prediction Logic ---

    # This block runs ONLY when the model is loaded AND a log file has been uploaded.
    if st.session_state.model_loaded and log_file:
        log_content = log_file.getvalue().decode('utf-8')

        with st.spinner("Parsing log file..."):
            logs_data = parse_bgl_logs(log_content)
            logs_df = pd.DataFrame(logs_data)
            st.session_state.logs_df = logs_df
            predictions = []
            with st.spinner("Analyzing logs..."):
                # create a progress bar for a better user experience
                progress_bar = st.progress(0)
                total_logs = len(logs_data)
                for i, log in enumerate(logs_data):
                    prediction = predict_anomaly(log, st.session_state.classifier, st.session_state.tokenizer)  #calling above function prediction anamoly
                    log.update(prediction)  # Add prediction results to the original log data
                    predictions.append(log)
                    # Update the progress bar
                    progress_bar.progress((i + 1) / total_logs)

            # Store the final predictions in the session state
            predictions_df = pd.DataFrame(predictions)
            st.session_state.predictions = predictions_df

    # --- Results Display Logic ---
    # This block runs ONLY when the predictions are available in the session state.
    if st.session_state.predictions is not None:
        st.header("Analysis Results")

        # --- Summary Metrics ---
        col1, col2, col3 = st.columns(3)
        total_logs = len(st.session_state.predictions)
        anomaly_count = st.session_state.predictions['is_anomaly'].sum()
        normal_count = total_logs - anomaly_count
        with col1:
            st.metric("Total Logs", total_logs)
        with col2:
            st.metric("Anomalies Detected", anomaly_count)
        with col3:
            st.metric("Normal Logs", normal_count)

        # --- Tabs for Detailed Views ---
        tab1, tab2, tab3 = st.tabs(["All Logs", "Anomalies Only", "Visualizations"])

        with tab1:
            # Display all logs with their predictions in a table
            st.dataframe(st.session_state.predictions[
                             ['epoch', 'component', 'category', 'severity', 'message', 'prediction',
                              'anomaly_probability']
                         ])

        with tab2:
            # Filter and display only the logs predicted as anomalies
            anomalies = st.session_state.predictions[st.session_state.predictions['is_anomaly'] == 1]
            if not anomalies.empty:
                st.dataframe(anomalies[
                                 ['epoch', 'component', 'category', 'severity', 'message', 'anomaly_probability']
                             ])
            else:
                st.info("No anomalies detected in the log file.")

        with tab3:
            # --- Visualizations ---
            st.subheader("Component-wise Anomaly Distribution")
            component_counts = st.session_state.predictions.groupby(['component', 'prediction']).size().reset_index(
                name='count')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='component', y='count', hue='prediction', data=component_counts, ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()  # Adjust plot to prevent labels from overlapping
            st.pyplot(fig)

            st.subheader("Anomaly Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.predictions['anomaly_probability'], bins=20, kde=True, ax=ax)
            plt.xlabel('Anomaly Probability')
            plt.ylabel('Count')
            st.pyplot(fig)

            st.subheader("Severity vs Prediction")
            severity_counts = st.session_state.predictions.groupby(['severity', 'prediction']).size().reset_index(
                name='count')
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='severity', y='count', hue='prediction', data=severity_counts, ax=ax)
            plt.tight_layout()
            st.pyplot(fig)


            # --- Export Results ---
            csv = st.session_state.predictions.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="log_analysis_results.csv",
                mime="text/csv"
            )


# ==============================================================================
# 6. SCRIPT ENTRY POINT
# ==============================================================================
# This is a standard Python convention. The code inside this block will only run
# when the script is executed directly (e.g., `python app.py`).
if __name__ == "__main__":
    main()


