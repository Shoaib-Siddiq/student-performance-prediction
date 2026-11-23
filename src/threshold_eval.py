"""Choose an early-warning threshold from training folds and measure held-out errors."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from src.data import load_dataset
from src.models import build_preprocessor, RANDOM_STATE


def main():
    frame = load_dataset()
    features = frame.drop(columns=['G1', 'G2', 'G3', 'passed'])
    failed = 1 - frame.passed
    train, test = train_test_split(np.arange(len(frame)), random_state=RANDOM_STATE,
                                  test_size=0.2, stratify=frame.passed)
    preprocessor, _ = build_preprocessor(features.iloc[train], [])
    model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=300, max_depth=12,
                          random_state=RANDOM_STATE, n_jobs=1))
    probabilities = cross_val_predict(model, features.iloc[train], failed.iloc[train],
                                      cv=5, method='predict_proba')[:, 1]
    grid = np.arange(0.05, 1, 0.05)
    chosen = max(grid, key=lambda t: f1_score(failed.iloc[train], probabilities >= t))
    model.fit(features.iloc[train], failed.iloc[train])
    predictions = model.predict_proba(features.iloc[test])[:, 1]
    results = {}
    for label, threshold in [('default', 0.5), ('training_cv_fail_f1', chosen)]:
        tn, fp, fn, tp = confusion_matrix(failed.iloc[test], predictions >= threshold).ravel()
        results[label] = dict(threshold=round(float(threshold), 3), correct_pass=int(tn),
                              false_alarm=int(fp), missed_fail=int(fn), detected_fail=int(tp))
    Path('reports/threshold_eval.json').write_text(json.dumps(results, indent=2)+'\n')
    print(results)


if __name__ == '__main__':
    main()
