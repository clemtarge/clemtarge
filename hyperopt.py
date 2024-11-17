
# https://wood-b.github.io/post/a-novices-guide-to-hyperparameter-optimization-at-scale/
# https://valohaichirpprod.blob.core.windows.net/papers/huawei.pdf
# https://docs.ray.io/en/latest/tune/examples/pbt_guide.html
# https://docs.ray.io/en/latest/tune/examples/tune-vanilla-pytorch-lightning.html

# https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks

# python opti.py > out_pbt.txt 2>&1

import comet_ml

import copy
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
import ray
from ray import train, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.air.integrations.comet import CometLoggerCallback
from ray.tune.search.hebo import HEBOSearch

import random
def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class HPO:
    """ Hyperparameter Optimization class """

    def __init__(self,
                 training_function,
                 default_config: dict,
                 metric: str,
                 mode: str,
                 training_function_kwargs: dict = {},
                 checkpoint_score_attribute: str = "",
                 checkpoint_score_order: str = "",
                 cpu: int = 1,
                 gpus_per_trial: int = 0,
                 data_dir: str = "",
                 comet_info: dict = {}):
        
        self.training_function = training_function
        self.training_function_kwargs = training_function_kwargs
        self.default_config = default_config
        self.metric = metric
        self.mode = mode
        self.checkpoint_score_attribute=checkpoint_score_attribute or metric
        self.checkpoint_score_order=checkpoint_score_order or mode
        self.cpu = cpu
        self.gpus_per_trial = gpus_per_trial
        self.data_dir = data_dir
        self.comet_info = comet_info

        self.results = {}
    
    def get_default_config(self):
        return copy.deepcopy(self.default_config)

    def getTune_with_resources(self,
                               #max_training_iteration: int,
                               training_function_kwargs: dict={}):
        return tune.with_resources(tune.with_parameters(self.training_function,
                                                        **training_function_kwargs,
                                                        ),
                                                        resources={"cpu": self.cpu, "gpu": self.gpus_per_trial})
    
    def getRunConfig(self,
                     name: str,
                     max_training_iteration: int):
        return train.RunConfig(name=name,
                               storage_path=self.data_dir,
                               checkpoint_config=train.CheckpointConfig(
                                   num_to_keep=3,
                                   checkpoint_score_attribute=self.checkpoint_score_attribute,
                                   checkpoint_score_order=self.checkpoint_score_order,
                                   ),
                               stop={"training_iteration": max_training_iteration},
                               callbacks=[CometLoggerCallback(**self.comet_info,
                                                              tags=[name])],
                               progress_reporter=CLIReporter(
                                   max_progress_rows=20,
                                   max_column_length=20,
                                   max_report_frequency=5,
                                   metric=self.metric,
                                   mode=self.mode,
                                   )
                               )
    
    def getCommonParametersTuneConfig(self):
        return {"metric": self.metric,
                "mode": self.mode,
                "max_concurrent_trials": 0,
                "trial_name_creator": lambda trial: f"{trial.trial_id}", # name of the Experiment
                "trial_dirname_creator": lambda trial: f"{trial.trial_id}"}
    
    def getTuner(self,
                 max_training_iteration: int,
                 training_function_kwargs: dict={},
                 param_space=None,
                 tune_config=None,
                 run_config=None):
        if param_space is None:
            param_space = self.get_default_config()
        if tune_config is None:
            tune_config = tune.TuneConfig(**self.getCommonParametersTuneConfig())
        if run_config is None:
            run_config = self.getRunConfig("default", max_training_iteration)

        return tune.Tuner(
            # self.getTune_with_resources(max_training_iteration),
            self.getTune_with_resources(training_function_kwargs=training_function_kwargs),
            param_space=param_space,
            tune_config=tune_config,
            run_config=run_config,
        )
    
    def tune_default(self,
                     max_training_iteration: int):
        """ Tune with default configuration """

        tuner = self.getTuner(max_training_iteration, training_function_kwargs=self.training_function_kwargs)
        results = tuner.fit()

    def tune_asha(self,
                  num_samples: int,
                  max_training_iteration: int,
                  param_space: dict,
                  reduction_factor: int,
                  n_best_to_keep: int=1):
        """ Tune with ASHA """

        tuner = self.getTuner(max_training_iteration=max_training_iteration,
                              param_space={**self.get_default_config(),
                                           **param_space,
                                           },
                              tune_config=tune.TuneConfig(**self.getCommonParametersTuneConfig(),
                                                          search_alg=HEBOSearch(metric=self.metric,
                                                                                mode=self.mode,
                                                                                random_state_seed=0), # pip install 'HEBO>=0.2.0'
                                                                                scheduler=ASHAScheduler(time_attr="training_iteration",
                                                                                                        grace_period=1,
                                                                                                        reduction_factor=reduction_factor),
                                                          num_samples=num_samples),
                              run_config=self.getRunConfig("asha", max_training_iteration))
        results = tuner.fit()

        self.results["asha"] = results.get_dataframe(filter_metric=self.metric, filter_mode=self.mode).sort_values(self.metric, ascending=self.mode=="min", ignore_index=True).iloc[:n_best_to_keep].filter(regex="config/.*").rename(columns=lambda x: x.split("/")[-1]).to_dict(orient="records")
        print(f"Best hyperparameters found:\n   {self.results['asha']}")

    def tune_pbt(self,
                 num_samples: int,
                 max_training_iteration: int,
                 param_space: dict,
                 perturbation_interval: int):
        """ Tune with PBT """

        if "asha" in self.results:
            configs = self.results["asha"]
        else:
            configs = [{}]

        for config in configs:
            print(f"\n------------------------------------ Start PBT with {config} ------------------------------------\n")

            tuner = self.getTuner(max_training_iteration=max_training_iteration,
                                  param_space={**self.get_default_config(), # order is important to overwrite default values
                                               **config,
                                               **param_space,
                                               },
                                  tune_config=tune.TuneConfig(**self.getCommonParametersTuneConfig(),
                                                              scheduler=PopulationBasedTraining(time_attr="training_iteration",
                                                                                                perturbation_interval=perturbation_interval,
                                                                                                hyperparam_mutations={
                                                                                                    "lr": tune.loguniform(1e-4, 5e-2),
                                                                                                    "batch_size": [16,32,64]
                                                                                                    }, # distribution for resampling
                                                                                                quantile_fraction=0.25,
                                                                                                resample_probability=0.25, # The probability of resampling from the original distribution
                                                                                                perturbation_factors=(1.2, 0.8),
                                                                                                log_config=True,
                                                                                                ),
                                                              num_samples=num_samples),
                                  run_config=self.getRunConfig("pbt", max_training_iteration))
            results = tuner.fit()

            best_result = results.get_best_result(metric=self.metric, mode=self.mode)
            print(f"Best hyperparameters found:\n   {best_result.config}")

    def tune_pbt_replay(self,
                        max_training_iteration: int,
                        pbt_policy_txt_path: str):
        """ Tune with PBT replay """
        import glob
        from ray.tune.schedulers import PopulationBasedTrainingReplay

        # Get a random replay policy from the experiment we just ran
        sample_pbt_trial_log = sorted(glob.glob(os.path.expanduser(pbt_policy_txt_path)))[0]
        replay = PopulationBasedTrainingReplay(sample_pbt_trial_log)

        tuner = self.getTuner(max_training_iteration=max_training_iteration,
                              tune_config=tune.TuneConfig(**self.getCommonParametersTuneConfig(), scheduler=replay),
                              run_config=self.getRunConfig("pbt_replay", max_training_iteration))
        results = tuner.fit()


if __name__ == '__main__':
    """ Example of hyperparameter optimization with PyTorch Lightning """

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from hyperopt_utils import data_dir, comet_info

    ray.init(num_gpus=1)#, _temp_dir=f"{data_dir}/ray/") ##### use symlink
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


    def train_model(config, num_epochs=4, data_dir=""):

        model = LightningMNISTClassifier(config=config, data_dir=data_dir)

        trainer = pl.Trainer(accelerator="cuda",
                            strategy="auto",
                            devices="auto",
                            #  logger=TensorBoardLogger(save_dir=f"{data_dir}/lightning_logs", name="", version="."),
                            logger=CometLogger(**comet_info),
                            max_epochs=num_epochs,
                            enable_progress_bar=False,
                            enable_model_summary=True,
                            callbacks=[TuneReportCheckpointCallback(metrics={"loss": "val_loss", "mean_accuracy": "val_accuracy"},
                                                                    filename="checkpoint.ckpt",
                                                                    on="validation_end")
                            ],
                            # profiler="simple"
        )

        # If `train.get_checkpoint()` is populated, then we are resuming from a checkpoint.
        checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                ckpt_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")
        else:
            ckpt_path = None

        trainer.fit(model, ckpt_path=ckpt_path)


    default_config = {"layer_1_size": 16,#8,
                      "layer_2_size": 32,#8,
                      "activation": "sigmoid",#"relu",
                      "lr": 1e-2,
                      "batch_size": 64
                      }

    param_space_architecture = {"layer_1_size": tune.choice([4, 8, 16, 32, 64]),
                                "layer_2_size": tune.choice([8, 16, 32, 64, 128]),
                                "activation": tune.choice(["relu", "sigmoid", "tanh", "leaky_relu"]),
                                }
    param_space_scheduler =  {"lr": tune.loguniform(1e-4, 5e-2),
                            "batch_size": tune.choice([16, 32, 64])
                            }

    # train_model(get_default_config(), num_epochs=5, data_dir=data_dir); exit()
    hpo = HPO(training_function=train_model,
              training_function_kwargs={"num_epochs": 5,
                                        "data_dir": data_dir},
              default_config=default_config,
              metric="loss", # "mean_accuracy"
              mode="min", # "max"
              checkpoint_score_attribute="mean_accuracy",
              checkpoint_score_order="max",
              cpu=1,
              gpus_per_trial=1,
              data_dir=data_dir,
              comet_info=comet_info)

    hpo.tune_default(3)
    # hpo.tune_asha(num_samples=3, max_training_iteration=3, param_space=param_space_architecture, reduction_factor=4, n_best_to_keep=1)
    # hpo.tune_pbt(num_samples=4, max_training_iteration=5, param_space=param_space_scheduler, perturbation_interval=2)
    # hpo.tune_pbt_replay(max_training_iteration=5, pbt_policy_txt_path="/tmp/ray/session_2024-08-13_14-51-33_008092_3855823/artifacts/2024-08-13_14-51-38/pbt/driver_artifacts/pbt_policy*.txt")
