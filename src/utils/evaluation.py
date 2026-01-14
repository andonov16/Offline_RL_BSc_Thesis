import torch
import numpy as np
import pandas as pd
from typing import Tuple, List

from src.experiments.BC_experiments import prepare_data
from src.normalization import NormalizationModule
from src.datasets import BCDataset

from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def print_model_eval_results(results: dict,
                             model_name: str = 'Replay Buffer') -> None:
    print('=' * 100)
    print(f'Model Evaluation results of {model_name}')

    for key, value in results.items():
        if isinstance(value, (list, tuple, np.ndarray)):
            print(f'{key}: {np.array(value)}')
        elif isinstance(value, dict):
            print(f'{key}:')
            for sub_key, sub_val in value.items():
                print(f'  {sub_key}: {sub_val}')
        else:
            print(f'{key}: {value}')


def evaluate_model_test_dataset(model: torch.nn.Module,
                                device: torch.device,
                                test_df: pd.DataFrame,
                                norm_technique_script: NormalizationModule,
                                selected_features: List[str]) -> Tuple[dict, np.array]:
    model.eval()
    model.to(device)

    # Prepare normalized and filtered test data
    X_test, y_test = prepare_data(df=test_df,
                                  selected_features=selected_features,
                                  norm_script=norm_technique_script)

    test_dataset = BCDataset(states=X_test, actions=y_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for states, labels in test_loader:
            states, labels = states.to(device), labels.to(device)
            logits = model(states)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    results = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

    return results, np.array(cm)