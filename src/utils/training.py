from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
import copy
import os
from torch.utils.data import DataLoader


def train_and_evaluate(train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                       model: torch.nn.Module,
                       early_stop_epoch_without_improvement: int = 3,
                       loss_function: callable = torch.nn.CrossEntropyLoss(), epochs=6, log_subfolder: str = 'logs',
                       tensorboard_subfolder: str = 'my_model',
                       show_progress: bool = True) -> tuple:
    """
        Trains and evaluates a given PyTorch model using the provided data loaders.

        Accepts:
            train_loader (DataLoader): the DataLoader for the training dataset
            val_loader (DataLoader): the DataLoader for the validation dataset
            optimizer (torch.optim.Optimizer): optimizer for training
            model (torch.nn.Module): PyTorch model to be trained
            early_stop_epoch_without_improvement (int, optional): number of epochs to wait without improvement
                before early stopping
            loss_function (callable, optional): the loss function to be used
            epochs (int, optional): Total number of training epochs
            log_subfolder (str, optional): Path to store the logs
            tensorboard_subfolder (str, optional): subfolder for TensorBoard logs
            show_progress (bool, optional): Whether to display a progress bar using tqdm

        Returns:
            tuple: best_model (torch.nn.Module), best_model_valid_accuracy (float).
        """
    # creates the log (sub)folders if necessary
    tensorboard_log_subfolder = os.path.join(log_subfolder, 'tensorboard')
    tensorboard_log_subfolder = os.path.join(tensorboard_log_subfolder, tensorboard_subfolder)
    if not os.path.exists(log_subfolder):
        os.makedirs(log_subfolder)
    if not os.path.exists(tensorboard_log_subfolder):
        os.makedirs(tensorboard_log_subfolder)
    log_writer = SummaryWriter(log_dir=tensorboard_log_subfolder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sample_batch = iter(train_loader)
    sample_states, _ = next(sample_batch)
    sample_actions = sample_states.to(device)

    log_writer.add_graph(model, sample_actions)

    best_model_path, best_val_loss, best_model_valid_accuracy, epochs_without_improvement = None, float('inf'), -1.0, 0
    train_losses, valid_losses = [], []

    itterator = range(epochs)
    if show_progress:
        itterator = tqdm(itterator, desc='Epochs')

    #main training loop
    for epoch in itterator:
        model.train()
        correct_train, total_train, train_loss = 0, 0, 0.0

        # update the weights
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        # log the train loss and accuracy using Tensorboard
        log_writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        log_writer.add_scalar("Accuracy/Train", train_accuracy, epoch)

        # evaluate the model after each completed training epoch on the validation set
        model.eval()
        correct_val, total_val, valid_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val
        avg_valid_loss = valid_loss / len(val_loader)

        # log the train loss and accuracy using Tensorboard
        log_writer.add_scalar("Loss/Valid", avg_valid_loss, epoch)
        log_writer.add_scalar("Accuracy/Valid", val_accuracy, epoch)

        # average loss per batch
        train_losses.append(train_loss / len(train_loader))
        valid_losses.append(valid_loss / len(val_loader))

        # early stopping:
        if valid_loss < best_val_loss:
            # best performing model here, save it:
            best_model_path = os.path.join(log_subfolder, f"curr_best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
            best_val_loss = valid_loss
            best_model_valid_accuracy = val_accuracy
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_epoch_without_improvement:
            break

    log_writer.flush()
    log_writer.close()

    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    return best_model, best_model_valid_accuracy
