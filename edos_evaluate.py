#!/usr/bin/env python

# load required packages
import sys
import os
import pandas as pd
from sklearn.metrics import f1_score

sys.stdout.write("Starting scoring program. \n\n")

# load submission and gold paths, which are passed as arguments as per the metadata file
submission_path = sys.argv[1]
gold_path = sys.argv[2]

# load submission
submission_df = pd.read_csv(submission_path)
sys.stdout.write("Loaded submission. \n")

# load gold standard data
gold_df = pd.read_csv(gold_path)
sys.stdout.write("Loaded gold standard data. \n\n")

# validate submission:
# correct columns exist
if "rewire_id" not in submission_df.columns:
    sys.exit('ERROR: Submission is missing Rewire ID column.')
if "label_pred_sexist" not in submission_df.columns:
    sys.exit('ERROR: Submission is missing label_pred_sexist column.')
if "label_pred_category" not in submission_df.columns:
    sys.exit('ERROR: Submission is missing label_pred_category column.')
# length matches gold standard data
if (len(submission_df) != len(gold_df)):
    sys.exit('ERROR: Number of entries in submission does not match number of entries in gold standard data. Are you submitting to the right task?')

# valid labels
unique_submission_labels_sexist = submission_df['label_pred_sexist'].unique()
unique_gold_labels_sexist = gold_df['label_sexist'].unique()
for i in unique_submission_labels_sexist:
    if i not in unique_gold_labels_sexist:
        sys.exit('ERROR: The column label_pred_sexist contains invalid label strings. Please see the Submission page for more information.')
unique_submission_labels_category = submission_df['label_pred_category'].unique()
unique_gold_labels_category = gold_df['label_category'].unique()
for i in unique_submission_labels_category:
    if i not in unique_gold_labels_category:
        sys.exit('ERROR: The column label_pred_category contains invalid label strings. Please see the Submission page for more information.')

sys.stdout.write("Submission contains correct column names. \n")
sys.stdout.write("Number of entries in submission matches number of entries in gold standard data. \n")
sys.stdout.write("Predicted labels are all valid strings. \n\n")

# sort submission and gold standard data by Rewire ID, so that labels match predictions
submission_df = submission_df.sort_values("rewire_id")
gold_df = gold_df.sort_values("rewire_id")

# calculate macro F1 score for the submission relative to the gold standard data
f1_sexist = f1_score(y_true = gold_df["label_sexist"], y_pred = submission_df["label_pred_sexist"], pos_label=None, average="macro")
f1_category = f1_score(y_true = gold_df["label_category"], y_pred = submission_df["label_pred_category"], average="macro")

sys.stdout.write("Submission evaluated successfully. \n")
sys.stdout.write("Macro F1 of TaskA: ")
sys.stdout.write(str(f1_sexist)+'\n')
sys.stdout.write("Macro F1 of TaskB: ")
sys.stdout.write(str(f1_category)+'\n')