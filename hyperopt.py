
import comet_ml
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from torch.nn import functional as F
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler

import random
def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def hyperoptimisation(training_function,
                      default_config: dict, # useful if not all parameters have to be optimized
                      metric: str,
                      mode: str,
                      n_trials: int,
                      param_config: dict,
                      n_startup_trials: int,
                      n_warmup_steps: int,
                      interval_steps: int,
                      n_jobs: int = 1,
                      seed: int = None,
                      training_function_kwargs: dict = {}):
    """ Perform hyperparameter optimization with Optuna """
    
    study = optuna.create_study(sampler=TPESampler(n_startup_trials=n_startup_trials,
                                                       n_ei_candidates=n_startup_trials,
                                                       seed=seed,
                                                       multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials,
                                                                   n_warmup_steps=n_warmup_steps,
                                                                   interval_steps=interval_steps,
                                                                   n_min_trials=1),
                                direction=mode,
                                )

    study.optimize(lambda trial: objective(trial, param_config, training_function, training_function_kwargs, metric, default_config),
                   n_trials=n_trials,
                   timeout=600,
                   n_jobs=n_jobs,
                   gc_after_trial=True,
                   show_progress_bar=True,
                   )
    
    results = study.trials_dataframe().sort_values("value", ascending=(mode=="minimize"))
    print("Study name:", study.study_name)
    print(results)
    results.to_csv("optuna_trials.csv", index=False)
    axs = optuna.visualization.matplotlib.plot_param_importances(study)
    fig = axs.figure
    fig.tight_layout()
    fig.savefig("optuna_param_importances.png")
    return


def objective(trial, param_config, training_function, training_function_kwargs, metric, default_config):

    param_space = {}
    for parameter_name, (type, space) in param_config.items():
        match type:
            case "categorical":
                param_space[parameter_name] = trial.suggest_categorical(parameter_name, space)
            case "loguniform":
                param_space[parameter_name] = trial.suggest_loguniform(parameter_name, *space)
            case "uniform":
                param_space[parameter_name] = trial.suggest_float(parameter_name, *space)
            case "int":
                param_space[parameter_name] = trial.suggest_int(parameter_name, *space)
            case _:
                raise ValueError(f"Unknown parameter type ({type}) for parameter {parameter_name}")

    # Merge default config with trial params (order is important to overwrite default values)
    param_space = {**default_config,
                   **param_space,
                   }
    
    # Return the monitored metric value
    return training_function(param_space, **training_function_kwargs, trial=trial, monitored_metric=metric)


if __name__ == '__main__':
    """ Example of hyperparameter optimization with PyTorch Lightning """

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from hyperopt_utils import data_dir, comet_info

    opj = os.path.join

    class LightningMNISTClassifier(pl.LightningModule):

        def __init__(self, config: dict, data_dir=None):
            set_seeds(0)
            super(LightningMNISTClassifier, self).__init__()

            self.data_dir = data_dir or os.getcwd()

            self.layer_1_size = config["layer_1_size"]
            self.layer_2_size = config["layer_2_size"]
            self.lr = config["lr"]
            self.batch_size = config["batch_size"]
            self.validation_step_outputs = []

            # mnist images are (1, 28, 28) (channels, width, height)
            self.layer_1 = torch.nn.Linear(28 * 28, self.layer_1_size)
            self.layer_2 = torch.nn.Linear(self.layer_1_size, self.layer_2_size)
            self.layer_3 = torch.nn.Linear(self.layer_2_size, 10)
            self.activation = {'relu': torch.nn.ReLU(),
                            'sigmoid': torch.nn.Sigmoid(),
                            'tanh': torch.nn.Tanh(),
                            'leaky_relu': torch.nn.LeakyReLU()}[config["activation"]]
            self.save_hyperparameters()

        def forward(self, x):
            batch_size, channels, width, height = x.size()
            x = x.view(batch_size, -1)

            x = self.layer_1(x)
            x = self.activation(x)

            x = self.layer_2(x)
            x = self.activation(x)

            x = self.layer_3(x)
            x = torch.log_softmax(x, dim=1)

            return x

        def cross_entropy_loss(self, logits, labels):
            return F.nll_loss(logits, labels)

        def accuracy(self, logits, labels):
            _, predicted = torch.max(logits.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / len(labels)
            return torch.tensor(accuracy)

        def on_train_epoch_start(self):
            set_seeds(0)

        # def on_fit_end(self):
        #     self.loggers[0].experiment.log_metric("epoch", self.current_epoch)
        #     self.loggers[0].log_hyperparams(self.hparams)

        def training_step(self, train_batch, batch_idx):
            x, y = train_batch
            logits = self.forward(x)
            loss = self.cross_entropy_loss(logits, y)
            accuracy = self.accuracy(logits, y)

            self.log("train_loss", loss)
            self.log("train_accuracy", accuracy)
            return loss

        def validation_step(self, val_batch, batch_idx):
            x, y = val_batch
            logits = self.forward(x)
            loss = self.cross_entropy_loss(logits, y)
            accuracy = self.accuracy(logits, y)
            out = {"val_loss": loss, "val_accuracy": accuracy}
            self.validation_step_outputs.append(out)
            return out

        def on_validation_epoch_end(self):
            outputs = self.validation_step_outputs
            avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
            self.log("val_loss", avg_loss)
            self.log("val_accuracy", avg_acc)
            self.validation_step_outputs.clear()  # free memory

        @staticmethod
        def download_data(data_dir):
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            return MNIST(data_dir, train=True, download=True, transform=transform)

        def prepare_data(self):
            mnist_train = self.download_data(self.data_dir)

            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=int(self.batch_size), num_workers=10)

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=int(self.batch_size), num_workers=10)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer


    def train_model(config, num_epochs, data_dir="", trial=None, monitored_metric="val_loss"):
        """ Train a model with the given config and return the monitored metric value """

        model = LightningMNISTClassifier(config=config, data_dir=data_dir)

        trainer = pl.Trainer(accelerator="cuda",
                            strategy="auto",
                            devices="auto",
                            #  logger=TensorBoardLogger(save_dir=f"{data_dir}/lightning_logs", name="", version="."),
                            logger=CometLogger(**comet_info),
                            max_epochs=num_epochs,
                            enable_progress_bar=True,
                            enable_model_summary=True,
                            callbacks=[PyTorchLightningPruningCallback(trial, monitor=monitored_metric)] if trial else [],
                            # profiler="simple"
        )

        trainer.fit(model)
        monitored_metric_value = trainer.callback_metrics[monitored_metric].item()
        return monitored_metric_value

    # Default config for the model
    default_config = {"layer_1_size": 16,
                      "layer_2_size": 32,
                      "activation": "sigmoid",
                      "lr": 1e-2,
                      "batch_size": 256,
                      }
    
    # Train the model with the default config
    # train_model(default_config, num_epochs=5, data_dir=data_dir)

    # Define the hyperparameter search space
    param_config = {"layer_1_size": ["categorical", [4, 8, 16, 32, 64]],
                    "layer_2_size": ["categorical", [8, 16, 32, 64, 128]],
                    "activation": ["categorical", ["relu", "sigmoid", "tanh", "leaky_relu"]],
                    "lr": ["loguniform", [1e-4, 1e-1]],
                    }

    # Perform hyperparameter optimization
    hyperoptimisation(training_function=train_model,
                      default_config=default_config, # useful if not all parameters have to be optimized
                      metric="val_loss",
                      mode="minimize",
                      n_trials=8,
                      param_config=param_config,
                      n_startup_trials=4,
                      n_warmup_steps=2,
                      interval_steps=2,
                      n_jobs=1,
                      #seed=0,
                      training_function_kwargs={"num_epochs": 6, "data_dir": data_dir})
    