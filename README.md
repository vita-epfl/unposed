# UnPOSed  
UnPOSed is an open-source toolbox for pose prediction/forecasting a sequence of human pose given an observed sequence, implemented in PyTorch.

<p float="left">
  Input pose<br/><img src="https://user-images.githubusercontent.com/33596552/138102745-f6b5c7a0-ee14-40ef-907f-b3ebb98ae08f.gif" alt="observation" width="500">

  Output pose and ground-truth<br/><img src="https://user-images.githubusercontent.com/33596552/138102754-5bef72df-ea48-4d17-a932-611293f0bc5a.gif" alt="prediction" width="500">  
</p>

# Overview 

The main parts of the library are as follows:

```
unposed
├── api
|   ├── preprocess.py                   -- script to run the preprocessor module
|   ├── train.py                        -- script to train the models, runs factory.trainer.py
│   ├── evaluate.py                     -- script to evaluate the models, runs factory.evaluator.py
|   └── generate_final_output.py        -- script to generate and save the outputs of the models, runs factory.output_generator.py
├── models                    
|   ├── st_trans/ST_Trans.py
|   ├── pgbig/pgbig.py
│   ├── history_repeats_itself/history_repeats_itself.py
│   ├── sts_gcn/sts_gcn.py
|   ├── zero_vel.py
│   ├── msr_gcn/msrgcn.py
|   ├── potr/potr.py
|   ├── pv_lstm.py
│   ├── disentangled.py
|   ├── derpof.py
|   ├── ...
├── losses
|   ├── pua_loss.py
│   ├── mpjpe.py           
|   ├── ...
```
The library has four important APIs to
- preprocess data 
- train the model
- evaluate a model quantitatively
- generate their outputs 

The details of how to use these API are described below. Two other important directories are models and losses. In these two directories, you can add any desired model and loss function and leverage all predefined functions of the library to train and test and compare in a fair manner.

Please check other directories (optimizers, mmetrics, schedulers, visualization, utils, etc.) for more abilites.

# Getting Started  
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch, evaluate your pretrained models, and produce basic visualizations.  
  
### Dependencies  
Make sure you have the following dependencies installed before proceeding:  
- Python 3.7+ distribution  
- PyTorch >= 1.7.0  
- CUDA >= 10.0.0 
- pip >= 21.3.1 
### Virtualenv  
You can create and activate virtual environment like below:  
```bash  
pip install --upgrade virtualenv

virtualenv -p python3.7 <venvname>  

source <venvname>/bin/activate  

pip install --upgrade pip
```  
### Requirements  

You can install all the required packages with:  
  
```bash  
pip install -r requirements.txt  
```  
Before moving forward, you need to install Hydra and know its basic functions to run different modules and APIs.  

## Hydra
Hydra is a framework for elegantly configuring complex applications with hierarchical structure.
For more information about Hydra, read their official page [documentation](https://hydra.cc/).

In order to have a better structure and understanding of our arguments, we use Hydra to  dynamically create a hierarchical configuration by composition and override it through config files and the command line.
If you have any issues and errors install hydra like below:
```bash
pip install hydra-core --upgrade
```
for more information about Hydra and modules please visit [here](ARGS_README.md#Hydra)

## MLFlow

We use MLflow in this library for tracking the training process. The features provided by MLFlow help users track their training process, set up experiments with multiple runs and compare runs with each other. Its clean, and organized UI helps users to better understand and track what the experiments. See more from MLFlow [here](https://mlflow.org/).

This part is not obligatory but you can use MLFlow by running the command below in the folder containing `mlruns` folder to track the training processes.
```bash
mlflow ui
```

# Datasets

We currently Support the following datasets:
- [Human3.6M](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
- [AMASS](https://amass.is.tue.mpg.de/en) from their official website..
- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website.


Please download the datasets and put them in a their specific folder. We will refer to this folder as `$DATASET_PATH` in the following sections.

# Models

We've tested the following models:
- [ST-Transformer](https://arxiv.org/abs/2304.06707)
- [PG-Big](https://arxiv.org/abs/2203.16051)
- [History Repeats Itself](https://arxiv.org/abs/2007.11755)
- [STS-GCN](https://arxiv.org/abs/2110.04573)
- [MSR-GCN](https://arxiv.org/abs/2108.07152)
- [PoTR](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/papers/Martinez-Gonzalez_Pose_Transformers_POTR_Human_Motion_Prediction_With_Non-Autoregressive_Transformers_ICCVW_2021_paper.pdf)
- [PV-LSTM](https://github.com/vita-epfl/bounding-box-prediction)
- [Disentangled](https://github.com/Armin-Saadat/pose-prediction-autoencoder)
- [DER-POF](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/html/Parsaeifard_Learning_Decoupled_Representations_for_Human_Pose_Forecasting_ICCVW_2021_paper.html)
- Zero-Vel

## Adding a Model

To add a new model, you need to follow the below steps:

- add the model file or files in the model directory
- add the model reference to the models.\_\_init\_\_.py
- add the model's required parameters to the configs/hydra/models. This step is necessary even if you don't have additional parameters
- if your model has new loss function which is not implemented in the library, you can add your loss function to the losses folder.

## Adding a Metric

To add a new metric, you need to follow the below steps:

- implement your metric function in the metrics.pose_metrics.py file
- add the model reference to the metrics.\_\_init\_\_.py
- add your metric to the configs/hydra/metrics.yml

# Preprocessing

We need to create clean static files to enhance dataloader and speed-up other parts.
To fulfill mentioned purpose, put the data in DATASET_PATH and run preprocessing api called `preprocess` like below:  

Example:  
```bash  
python -m api.preprocess \
    dataset=human3.6m \
    annotated_data_path=$DATASET_PATH \
    data_type=test
```  
See [here](ARGS_README.md#preprocessing) for more details about preprocessing arguments.
This process should be repeated for training, validation and test set. This is a one-time use api and later you just use the saved jsonl files.
  
# Training

Given the preprocessed data, train models from scratch:
```bash  
python -m api.train model=st_trans \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=50 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=human3.6m \
    model.n_major_joints=22 \
    model.loss.nJ=32
```  
DATASET_TRAIN_PATH and DATASET_VALIDATION_PATH refer to the preprocessed json files.

**NOTE**: You can see more commands for training models [here](training_commands.md).

Provide **validation_dataset** to adjust learning-rate and report metrics on validation-dataset as well.

See [here](ARGS_README.md#training) for more details about training arguments.


# Evaluation

You can evaluate any pretrained model with:
```bash
python -m api.evaluate model=st_trans \
          dataset=$DATASET_TEST_PATH \
          load_path=$MODEL_CHECKPOINT \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_h36_testing=true
```
in case of non-trainable model, run:
```bash  
python -m api.evaluate model=zero_vel \
          dataset=$DATASET_TEST_PATH \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_h36_testing=true
```  
See [here](ARGS_README.md#evaluation) for more details about evaluation arguments.



## Epistemic Uncertainty

You can also evaluate the epistemic uncertainty of the models using the approach presented in the paper. For the ease of use, we have provided the trained EpU model in the release section.
Using the trained EpU model, one can evaluate the epistemic uncertainty of the pose prediction model:
```bash
python -m api.evaluate model=st_trans \
          dataset=$DATASET_TEST_PATH \
          load_path=$MODEL_CHECKPOINT \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_testing=true \
          data.is_h36_testing=true \
          dataset_name=human3.6m   \
          eval_epu=true \
          epu_model_path=$EPU_MODEL
```


# Generating Outputs

Generate and save the predicted future poses:
```bash
python -m api.generate_final_output model=st_trans \
          dataset=$DATASET_PATH \
          load_path=$MODEL_CHECKPOINT \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_h36_testing=true \
          save_dir=$OUTPUT_PATH
```  
See [here](ARGS_README.md#generating-outputs) for more details about prediction arguments.



# Work in Progress
This repository is being updated so please stay tuned!   
