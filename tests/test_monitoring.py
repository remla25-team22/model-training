

import pandas as pd
import os
import sys
import joblib
from memory_profiler import memory_usage
import pytest



def run_inference(model, vectorizer, cleaned_texts):
    x = vectorizer.transform(cleaned_texts)
    _ = model.predict(x)


@pytest.mark.ml_test("MON-6")   # MON-6: The model has not experienced a dramatic or slow-leak regressions in training speed, serving latency, throughput, or RAM usage:
def test_inference_memory_under_500mb():
    model = joblib.load("models/c2_model.pkl")
    vectorizer = joblib.load("models/c1_BoW.pkl")

    df = pd.read_csv("data/preprocessed/test.csv")  
    cleaned_texts = df["cleaned"].astype(str).tolist()

    mem_usage = memory_usage((run_inference, (model, vectorizer, cleaned_texts)), max_iterations=1)
    print("usage blya ", mem_usage)
    peak_memory = max(mem_usage)

    assert peak_memory < 500, f"Inference used too much memory: {peak_memory:.2f} MB"
