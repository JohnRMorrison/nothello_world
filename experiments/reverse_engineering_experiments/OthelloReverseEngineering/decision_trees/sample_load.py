"""
Sample loader for decision tree results
"""

from pathlib import Path
import gzip
import pickle
import sys
import dtypes
from dtypes import DecisionTreeResults, BinaryDecisionTreeResults


BASE_DIR = Path(__file__).resolve().parent


# =========================
# Continuous feature regression trees
# =========================
for cont_layer in range(1, 8):
    cont_path = BASE_DIR / "continuous_features" / "results" / f"layer_{cont_layer}_trees.pkl.gz"
    with gzip.open(cont_path, "rb") as f:
        cont_feature_regressors = pickle.load(f)

    assert isinstance(cont_feature_regressors, list)
    assert len(cont_feature_regressors) == 2048
    assert isinstance(cont_feature_regressors[0], DecisionTreeResults)
    print(sum(hasattr(regressor.tree, 'tree_') for regressor in cont_feature_regressors))



# =========================
# Ground truth feature classification trees
# =========================
for gt_class_layer in range(8):
    gt_class_path = (
        BASE_DIR
        / "ground_truth_features"
        / "classification"
        / "results"
        / f"layer_{gt_class_layer}_trees.pkl.gz"
    )

    with gzip.open(gt_class_path, "rb") as f:
        gt_feature_classifiers = pickle.load(f)
        
    assert isinstance(gt_feature_classifiers, list)
    assert len(gt_feature_classifiers) == 2048
    assert isinstance(gt_feature_classifiers[0], BinaryDecisionTreeResults)
    print(sum(hasattr(classifier.tree, 'tree_') for classifier in gt_feature_classifiers))
    for idx, classifier in enumerate(gt_feature_classifiers):
        if not hasattr(classifier.tree, 'tree_'):
            print(f"L{gt_class_layer}N{idx} tree not fitted")

# =========================
# Ground truth features regression trees
# =========================
for layer in range(8):
    gt_reg_layer = layer
    gt_reg_path = (
        BASE_DIR
        / "ground_truth_features"
        / "regression"
        / "results"
        / f"layer_{gt_reg_layer}_trees.pkl.gz"
    )
    with gzip.open(gt_reg_path, "rb") as f:
        gt_feature_regressors = pickle.load(f)

    assert isinstance(gt_feature_regressors, list)
    assert len(gt_feature_regressors) == 2048
    assert isinstance(gt_feature_regressors[0], DecisionTreeResults)
    print(sum(hasattr(regressor.tree, 'tree_') for regressor in gt_feature_regressors))


