# Sequential Recommender Systems Reproducibility Analysis
This is the code for the paper Sequential Recommender Systems Reproducibility Analysis submitted at RecSys24.

**EasyRec** is a versatile Python library carefully designed to streamline the process of configuring and building Sequential Recommender Systems models using the power and robust capabilites of PyTorch Lightning and PyTorch models.

## 🌟 Key Features

1. **User-Friendly Configuration:** EasyRec revolutionizes the way you work with Sequential Recommender Systems by providing a seamless configuration-based interface. All your settings can be easily defined in YAML files, making it effortless to customize and fine-tune your experiments.

2. **Three Essential Utilities:**
   - 📊 **Data:** EasyRec simplifies data handling, allowing you to load and preprocess data effortlessly.
   - 📝 **Experiments:** Define, track, and save experiments with unique IDs to prevent duplication. Keep your work organized and accessible.
   - ⚙️ **Torch Integration:** Seamlessly integrate PyTorch models, train, test, and save your models with minimal effort. EasyRec handles the heavy lifting, so you can focus on innovation.

## 📁 Important Files and Folders in the Project

Below is an outline of key files and folders you'll find in this project, along with their purposes:


### Utilities
1. **easy_exp**
    - Includes functions for experiment handling, such as creating experiment IDs, saving and loading experiments, and managing experiment logs.
2. **easy_rec**
    - Includes functions for data pre-processing and dataloaders creation
    - Defines Sequential Recommendation Models
    - Contains IR metrics, such as NDCG, Recall, and MRR
    - Provides losses for model's training and evaluation
3. **easy_torch**
    - Includes functions for metrics, loading models, and creating trainers in PyTorch Lightning.
    - Defines steps, loss, optimizer, and other parameters to use.
    - Sets callbacks and dataloaders.
    - Also includes utilities for training and testing the model, as well as saving and reading logs.

### Folders

1. **cfg**
    - Contains the configurations used in the testing phase of this repo: these are YAML files nested inside each other. The main one is config_rec.yaml, where you specify the name of the experiment.
      - data_cfg: loading and preprocessing parameters of the dataset.
      - model: optimizer, metrics and name of the model used.
      - trainer_params_cfg: accelerator, number of epochs, logger to save files.
      - loader_params_cfg: batch size, number of workers, number of negatives.
      - emission_tracker: configuration for [CodeCarbon](https://codecarbon.io/).
      - flops_profiler: configuration for [DeepSpeed])(https://deepspeed.readthedocs.io/en/latest/index.html).
      - rec_models: specific configurations for each model.
2. **ntb**
    - Houses a notebook for training and testing a Sequential Recommender System on a dataset.
3.  **out**
    - Includes metrics and energy consumption for each experiments. Also includes the saved parameters of the best model per single run.


## How to run an experiment

To run our code follow the next steps:

``mkdir easy_lightning``
- Download this [repo](https://anonymous.4open.science/r/easy_lightning-B93D). Since an anonymous repo can not be cloned, you have to download it on your machine manually. Insert the zip in the folder `easy_lightning` and unzip it.


``cd easy_lightning && unzip easy_lightning-B93D.zip``
- Install the repo just downloaded. Go in the parent directory of `easy_lightning`. If `easy_lightning` is in Desktop, you have to stay in Desktop;

``cd .. && pip3 install --upgrade --force-reinstall easy_lightning/ > /dev/null 2>&1``

- Create the folder `recsys_posneg`.

``mkdir recsys_repro``
- Download **this repo**. As before, it can't be cloned. Insert the zip in the folder `recsys_repro`.

- Select the right directory;

``cd recsys_repro && unzip recsys_repro_conf-D117``

- Install the necessary requirements.

``pip3 install -r requirements.txt``
- Download the data;

``cd ntb && bash download_data.bash``
- Run a simple experiment. By default, these files are set to run SASRec on ml-100k dataset.

``python3 main.py``
