from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


@dataclass
class DecisionTreeResults:
    """Results for a single square's decision tree."""

    layer: int
    neuron: int
    tree: DecisionTreeRegressor
    train_R2: float
    train_MSE: float
    test_R2: float
    test_MSE: float


@dataclass
class BinaryDecisionTreeResults:
    """Results for a single square's decision tree."""

    layer: int
    neuron: int
    tree: DecisionTreeClassifier
    train_precision: float
    train_recall: float
    train_accuracy: float
    train_F1: float
    test_precision: float
    test_recall: float
    test_accuracy: float
    test_F1: float