import pandas as pd
import sys
import os

from models import model_evaluation
from preprocess.fft_channel import process_fft_channel
from preprocess.fft_total import process_fft_total
from preprocess.td_channel import process_td_channel
from preprocess.td_total import process_td_total

indices_train = "./data/train_set.csv"
indices_test = "./data/test_set.csv"

train_meta_df = pd.read_csv(indices_train, sep="\t")
test_meta_df = pd.read_csv(indices_test, sep="\t")

os.makedirs("./logging", exist_ok=True)
sys.stdout = open("./logging/out_model.txt", "w")
sys.stderr = open("./logging/err_model.txt", "w")

def model_final_eval(data_path, preprocess_func):
    X_train, y_train, X_test, y_test = preprocess_func(data_path, train_meta_df, test_meta_df)

    model_evaluation(X_train, y_train, X_test, y_test)


model_final_eval("./data/csv/fft_channel_fill.csv", process_fft_channel)
model_final_eval("./data/csv/fft_channel_nofill.csv", process_fft_channel)
model_final_eval("./data/csv/fft_total_fill.csv", process_fft_total)
model_final_eval("./data/csv/fft_total_nofill.csv", process_fft_total)
model_final_eval("./data/csv/td_channel_fill.csv", process_td_channel)
model_final_eval("./data/csv/td_channel_nofill.csv", process_td_channel)
model_final_eval("./data/csv/td_total_fill.csv", process_td_total)
model_final_eval("./data/csv/td_total_nofill.csv", process_td_total)

sys.stdout.close()
sys.stderr.close()
