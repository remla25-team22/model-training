
import json
import shutil
from pathlib import Path
import sys
import os
import pytest
import subprocess
import pickle
from lib_ml.preprocess import clean_review

from model_training import config
from model_training import train
from model_training import evaluate
from model_training import data_prep

import pandas as pd
import joblib
import numpy as np


IGNORED = shutil.ignore_patterns(
    ".git", ".dvc",  
    "data", "models", 
    "__pycache__", "*.pyc", ".pytest_cache"
)

RAW_WORDS = {
    "good": "fine",
    "excellent": "great",
    "bad": "poor"
}
CLEANED_MAP = {clean_review(k): clean_review(v) for k, v in RAW_WORDS.items()}

MIN_INVARIANCE = 0.85

@pytest.mark.ml_test("INF-1")  # Infra-1: Training is reproducible
def test_determinism(tmp_path, monkeypatch):
    seeds = [0, 12, 42, 100, 1000]
    min_acc = 0.7

    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    data_dir.mkdir()
    model_dir.mkdir()
    reports_dir.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "REPORT_DIR", reports_dir)

    raw_data_src = Path("data/raw/a1_RestaurantReviews_HistoricDump.tsv")
    shared_raw_dir = tmp_path / "raw"
    shared_raw_dir.mkdir()
    shutil.copy2(raw_data_src, shared_raw_dir / raw_data_src.name)

    results = []

    for seed in seeds:
        monkeypatch.setattr(config, "RANDOM_SEED", seed)
        for p in data_dir.iterdir():
            if p.name != "raw":
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
        for p in model_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()

        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared_raw_dir / raw_data_src.name, raw_dir / raw_data_src.name)

        data_prep.main()
        train.main()
        evaluate.main()

        with open(reports_dir / "metrics.json", "r") as f:
            acc = json.load(f)["accuracy"]
            results.append(acc > min_acc)

    assert all(results), f"Some seeds failed: {[(s, r) for s, r in zip(seeds, results) if not r]}"




def _copy_project_skeleton(src: Path, dst: Path) -> None:
    """
    Copy just the files the pipeline needs (code + dvc files), skipping any
    real data or models so the test cannot overwrite them.
    """
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=IGNORED)


def _create_mini_dataset(raw_dir: Path) -> None:
    """
    Drop a 6-row deterministic sample so the pipeline runs in <1 s.
    """
    sample = (
        "Review\tLiked\n"
        "Amazing food!\t1\n"
        "Terrible service.\t0\n"
        "Loved the ambience.\t1\n"
        "Too salty.\t0\n"
        "Will come again!\t1\n"
        "Never returning.\t0\n"
        "Very rude staff.\t0\n"
        "Burger was bad quality.\t0\n"
        "Amazing atmosphere.\t1\n"
        "Cheese had a bad quality.\t0\n"
        "Amazing food!\t1\n"
        "Terrible service.\t0\n"
        "Loved the ambience.\t1\n"
        "Too salty.\t0\n"
        "Will return again!\t1\n"
        "Never returning.\t0\n"
        "Very nice staff.\t1\n"
        "Burger was bad quality.\t0\n"

    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "a1_RestaurantReviews_HistoricDump.tsv").write_text(sample, encoding="utf-8")


@pytest.mark.ml_test("INF-3")  # Infra-3: full pipeline integration test
def test_full_dvc_pipeline(tmp_path: Path) -> None:
    """
    Infra 3: Verify that every stage (prepare → train → evaluate) still
    executes end-to-end without crashing or losing its outputs.
    """
    project_root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "sandbox"
    _copy_project_skeleton(project_root, workspace)

    _create_mini_dataset(workspace / "data" / "raw")

    subprocess.run(["dvc", "init", "--no-scm", "-q"], cwd=workspace, check=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(workspace / "src")

    subprocess.run(["dvc", "repro", "-q"], cwd=workspace, check=True, env=env)

    assert (workspace / "data" / "preprocessed" / "train.csv").stat().st_size > 0
    assert (workspace / "models" / "c2_model.pkl").exists()
    assert (workspace / "reports" / "metrics.json").exists()


@pytest.mark.ml_test("INF-4")   # Infra-4: Model quality is validated before attempting to serve it
def test_synonym_swap_invariance():

    df = pd.read_csv('data/preprocessed/test.csv')
    texts = df['cleaned'].tolist()

    def contains_word(text, words):
        tokens = text.split()
        return any(w in tokens for w in words)

    selected = [t for t in texts if contains_word(t, CLEANED_MAP.keys())]
    assert selected, "No examples with target words found in test set!"

    # Create mutated texts by swapping cleaned words with their cleaned synonyms
    mutated = []
    for text in selected:
        tokens = text.split()
        swapped = [CLEANED_MAP.get(tok, tok) for tok in tokens]
        mutated.append(" ".join(swapped))

    with open('models/c1_BoW.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    model = joblib.load('models/c2_model.pkl')

    X_orig = vectorizer.transform(selected)
    X_mut = vectorizer.transform(mutated)
    preds_orig = model.predict(X_orig)
    preds_mut = model.predict(X_mut)

    invariance = np.mean(preds_orig == preds_mut)
    assert invariance >= MIN_INVARIANCE, (
        f"Synonym swap invariance too low: {invariance:.2%} < {MIN_INVARIANCE:.2%}"
    )