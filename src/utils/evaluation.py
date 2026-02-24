import torch
import os
import numpy as np
import pandas as pd
from typing import Tuple, List

from src.experiments.bc_experiments import prepare_data
from src.normalization import NormalizationModule
from src.datasets import BCDataset
from src.utils.plotting import plot_confusion_matrix_heatmap

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
                                selected_features: List[str],
                                apply_softmax: bool = True) -> Tuple[dict, np.array]:
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
            if apply_softmax:
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            else:
                preds = torch.argmax(logits, dim=1)
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


def evaluate_BC_models_on_test_dataset(
    dataset_type:str = 'replay_buffer',
    norm_names: List[str] = List[str],
    selected_features: List[str] = List[str],
    plots_dir: str = '/plots',
    torch_device: torch.device = 'cpu',
) -> pd.DataFrame:
    if dataset_type == 'replay_buffer':
        dataset_short_name = 'rb'
    elif dataset_type == 'final_policy':
        dataset_short_name = 'fp'
    else:
        dataset_short_name = ''

    test_df = (
        pd.read_parquet(f'../../data/{dataset_type}_episodes/{dataset_short_name}_test.parquet')
          .drop(columns=['done', 'episode', 'reward'])
    )
    all_results = {}

    for norm_name in norm_names:
        print('-'*100)
        print(f"\nEvaluating normalization: {norm_name}")

        if norm_name == 'raw':
            normalization_technique = None
        else:
            normalization_technique = torch.jit.load(
                f'../../models/BC/{dataset_type}/normalization/{norm_name}_normalization.pt'
            )

        model = torch.jit.load(
            f'../../models/BC/{dataset_type}/BC_{norm_name}.pt'
        )

        # Evaluate
        report_dict, confusion_matrix = evaluate_model_test_dataset(
            model=model,
            device=torch_device,
            test_df=test_df,
            norm_technique_script=normalization_technique,
            selected_features=selected_features
        )

        if dataset_type == 'replay_buffer':
            fig_dataset_name = 'Replay Buffer'
        elif dataset_type == 'final_policy':
            fig_dataset_name = 'Final Policy'
        else:
            fig_dataset_name = 'Unknown'

        # Plot + save confusion matrix
        fig = plot_confusion_matrix_heatmap(
            confusion_matrix=confusion_matrix,
            model_name=f'BC {fig_dataset_name} ({norm_name})',
            f_size=(5, 5)
        )

        fig_path = os.path.join(
            plots_dir,
            f'confusion_matrix_bc_{dataset_type}_{norm_name}.png'
        )
        fig.savefig(fig_path, bbox_inches='tight')

        # Store results
        flat_metrics = {
            'accuracy': report_dict['accuracy'],
            'balanced_accuracy': report_dict['balanced_accuracy'],
            'precision_macro': report_dict['precision_macro'],
            'recall_macro': report_dict['recall_macro'],
            'f1_macro': report_dict['f1_macro'],
            'precision_weighted': report_dict['precision_weighted'],
            'recall_weighted': report_dict['recall_weighted'],
            'f1_weighted': report_dict['f1_weighted']
        }

        all_results[norm_name] = flat_metrics

        #print_model_eval_results(
        #    report_dict,
        #    model_name=f'BC Final Policy ({norm_name})'
        #)

    summary_df = pd.DataFrame.from_dict(all_results, orient='index')
    summary_df.index.name = 'normalization'
    summary_df = summary_df.sort_values(by='balanced_accuracy', ascending=False)

    return summary_df