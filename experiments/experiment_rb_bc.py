import os.path
from itertools import product

from src.utils.data_loading import *
from src.utils.config_managing import *
from src.utils.training import *
from src.behavior_cloning import BC

# import the corresponding config dict
bc_config_dict = load_bc_config_file('../config/rb_bc_config.json')

# get Data Loaders
rb_observations, rb_next_observations, rb_actions, rb_rewards, rb_dones = load_data()

train_loader, test_loader, valid_loader = get_BC_data_loaders(observations=rb_observations,
                                                              actions=rb_actions.flatten(),
                                                              train=bc_config_dict['train_ratio'],
                                                              test=bc_config_dict['test_ratio'],
                                                              validation=bc_config_dict['valid_ratio'],
                                                              batch_size=bc_config_dict['batch_size'],
                                                              seed=bc_config_dict['seed'])

# generate all possible hyperparameter combinations using grid search
for key, value in bc_config_dict.items():
    if not isinstance(value, list):
        bc_config_dict[key] = [bc_config_dict[key]]
keys, values = zip(*bc_config_dict.items())
grid_search_itters = [dict(zip(keys, v)) for v in product(*values)]

# apply grid search to find the best hyperparameter combination (save the best hyperparams and accuracy)
best_model, best_valid_accuracy, best_hyperparams = None, -1, None
for hyperparams in tqdm(grid_search_itters, desc='Grid Search'):
    BC_model = BC(input_neurons=hyperparams['input_neurons'],
                  hidden_neurons=hyperparams['hidden_neurons'],
                  num_hidden_layers=hyperparams['num_hidden_layers'],
                  out_neurons=hyperparams['out_neurons'],
                  activation_function=torch.nn.ReLU())

    optimizer = torch.optim.Adam(BC_model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])

    BC_model, curr_valid_accuracy = train_and_evaluate(train_loader=train_loader,
                                                       val_loader=valid_loader,
                                                       model=BC_model,
                                                       optimizer=optimizer,
                                                       early_stop_epoch_without_improvement=hyperparams[
                                                           'early_stop_epoch_without_improvement'],
                                                       loss_function=torch.nn.CrossEntropyLoss(),
                                                       epochs=hyperparams['epochs'],
                                                       log_subfolder=hyperparams['log_subfolder'],
                                                       tensorboard_subfolder=str(
                                                           f'lr={hyperparams["lr"]}_wd={hyperparams["weight_decay"]}_hidden_neurons={hyperparams["hidden_neurons"]}_num_hidden_layers={hyperparams["num_hidden_layers"]}'),
                                                       show_progress=False)

    # update the best model if the current configuration has higher validation accuracy
    if curr_valid_accuracy > best_valid_accuracy:
        best_hyperparams = hyperparams
        best_model = BC_model
        best_valid_accuracy = curr_valid_accuracy

# train the final model with the best hyperparameter combination
best_BC_model, best_valid_accuracy = train_and_evaluate(train_loader=train_loader,
                                                        val_loader=valid_loader,
                                                        model=BC_model,
                                                        optimizer=optimizer,
                                                        early_stop_epoch_without_improvement=best_hyperparams['early_stop_epoch_without_improvement'],
                                                        loss_function=torch.nn.CrossEntropyLoss(),
                                                        epochs=100,
                                                        log_subfolder=hyperparams['log_subfolder'],
                                                        tensorboard_subfolder=str(
                                                            f'lr={best_hyperparams["lr"]}_wd={best_hyperparams["weight_decay"]}_hidden_neurons={hyperparams["hidden_neurons"]}_num_hidden_layers={hyperparams["num_hidden_layers"]}'),
                                                        show_progress=False)

# compute the test accuracy
best_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = best_model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

best_test_accuracy = correct / total

# save the best model as a file
model_save_path = "../models"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
model_save_path = os.path.join(model_save_path, 'rb_BC_best_model.pth')

torch.save(best_model.state_dict(), model_save_path)

# save the best hyperparameters and accuracies
log_file_path = os.path.join(best_hyperparams['log_subfolder'], "best_hyperparams.txt")

with open(log_file_path, "w") as f:
    f.write(f"Best Hyperparameters: {best_hyperparams}\n")
    f.write(f"Best Validation Accuracy: {best_valid_accuracy:.4f}\n")
    f.write(f"Best Test Accuracy: {best_test_accuracy:.4f}\n")

print(f"Best Hyperparameters: {best_hyperparams}")
print(f"Best Validation Accuracy: {best_valid_accuracy}")
print(f"Best Test Accuracy: {best_test_accuracy}")
