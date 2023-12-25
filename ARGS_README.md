# Arguments
This is a description to provide details about arguments of Posepred API.
Pospred is an open-source toolbox for pose prediction in PyTorch. Posepred is a library that provides a unified interface to train, evaluate, and visualize the models. The library has 5 important APIs. The details of how to use these API are described below. Two other important directories are models and losses. In these two directories, you can add any desired model and loss function and leverage all predefined functions of the library to train and test and compare in a fair manner. 

# Hydra
```
posepred
├── configs
│   ├── hydra                     
|      ├── data
|         └── main.yaml                 -- main config file for data module (Essentially responsible for creating dataloader)             
|      ├── model
|         ├── common.yaml                 -- share config file for all models
|         ├── st_trans.yaml                 
│         ├── history_repeats_itself.yaml           
|         ├── sts_gcn.yaml
|         ├── ...              
|      ├── optimizer
|         ├── adam.yaml                 -- config file for adam optimizer
|         ├── sgd.yaml                  -- config file for stochastic gradient descent optimizer
|         ├── ...   
|      ├── scheduler
|         ├── reduce_lr_on_plateau.yaml -- config file for reducing learning_rate on plateau technique arguments
|         ├── step_lr.yaml              -- config file for step of scheduler arguments                               
|         ├── ...   
|      ├── visualize.yaml               -- config file for visualizer API arguments
|      ├── evaluate.yaml                -- config file for evaluate API arguments 
|      ├── preprocess.yaml              -- config file for preprocess API arguments
|      ├── train.yaml                   -- config file for train API arguments
|      ├── generate_output.yaml         -- config file for generate_output API arguments       
|      |── metrics.yaml                 -- config file for metrics
|      |── shared.yaml                  -- config file for shared arguments for the apis
|                    
└─── logging.conf                 -- logging configurations
```
Now we will precisely explain each module.
#### data
Location: 'configs/hydra/data'

`main.yaml`:
```
Mandatory arguments:
keypoint_dim:               Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
model_pose_format:          Used data format for pose dataset (str)
metric_pose_format:         Used data format for metrics if pose dataset is used. If no value is specified it'll use the model_pose_format's value (str)
is_h36_testing:             Set True to configure the dataloader for testing huamn3.6m (bool)
is_testing:                 Set True to configure the dataloader for testing (bool) (default: False)
batch_size:                 Indicates size of batch size (int) (default: 256)
shuffle:                    Indicates shuffling the data in dataloader (bool) (default: False)
pin_memory:                 Using pin memory or not in dataloader (bool) (default: False)  
num_workers:                Number of workers (int)
len_observed:               Number of frames to observe (int)
len_future:                 Number of frames to predict(int)

optional arguments: 
seq_rate:                   The gap between start of two adjacent sequences (1 means no gap) (int) (default: 2) (only used for pose data_loader) 
frame_rate:                 The gap between two frames (1 means no gap) (int) (default: 2) (only used for pose data_loader) 
```
#### model
Folder Location: 'configs/hydra/model'

**Available model names for Apis:** st_trans, msr_gcn, pgbig, sts_gcn, history_repeats_itself, potr, pv_lstm, derpof, disentangled, zero_vel

`common.yaml`:
```
Mandatory arguments:
keypoint_dim:               Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
pred_frames_num:            Number of frames to predict, obligatory when ground-truth is not available (int)
obs_frames_num:	            Number of frames to observe (int)
mean_pose:
std_pose:
device:                     Choose either 'cpu' or 'cuda' (str)
```

`<model-name>.yaml`:

For each model you implement, you should provide a yaml file to configure its argumants.
```
Mandatory arguments:
type:                       Name of the model (str)
loss.type:	            Name of the loss function (str)

optional arguments: 
Every specific argument required for your model!
```

#### optimizer
Folder Location: 'configs/hydra/optimizer'

`adam.yaml`
```
type                        type=adam for adam optimizer (str)
lr                          learning rate (float) (default=0.001)
weight_decay                weight decay coefficient (default: 1e-5)
```
`adamw.yaml`
```
type                        type=adamw for adamw optimizer (str)
lr                          learning rate (float) (default=0.001)
betas                       coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
weight_decay                weight decay coefficient (default: 1e-5)
```
`sam.yaml`
```
type                        type=sam for sharpness aware minimization (str)
lr                          learning rate (float) (default=0.001)
weight_decay                weight decay coefficient (default: 1e-5)
```
`sgd.yaml`
```
type                        type=sgd for stochastic gradient descent (str)
lr                          learning rate (float) (default=0.001)
momentum                    momentum factor in sgd optimizer (float) (default=0)
dampening                   dampening for momentum in sgd optimizer (float) (default=0)
weight_decay                weight decay coefficient (default: 1e-5)
nesterov                    enables Nesterov momentum (bool) (default=False)
```

#### scheduler
Folder Location: 'configs/hydra/scheduler'

`multi_step_lr.yaml`
```
type                        type=multi_step_lr to use this technique
step_size                   List of epoch indices. Must be increasing.
gamma                       Multiplicative factor of learning rate decay. (float) (default=0.4)
```
`reduce_lr_on_plateau.yaml`
```
type                        type=reduce_lr_on_plateau to use this technique (str)
mode                        One of `min`, `max`. In `min` mode, lr will be reduced when the quantity monitored has stopped
                            decreasing; in `max` mode it will be reduced when the quantity monitored has stopped increasing (str) (default=min)     
factor                      actor by which the learning rate will be reduced. new_lr = lr * factor (float) (default=0.5)
patience                    Number of epochs with no improvement after which learning rate will be reduced. (int) (default=20)
threshold                   Threshold for measuring the new optimum, to only focus on significant changes (float) (default=le-3)
verbose                     If True, prints a message to stdout for each update. (bool) (defaulTrue)
```
`step_lr.yaml`
```
type                        type=step_lr to use this technique
step_size                   Period of learning rate decay (int) (default=50)
gamma                       Multiplicative factor of learning rate decay. (float) (default=0.5)
last_epoch                  The index of last epoch (int) (default=-1)
verbose                     If True, prints a message to stdout for each update (bool) (default=False)
```
#### metrics
File Location: 'configs/hydra/metrics.yaml'

`metrics.yaml`:
```
pose_metrics:               List which metrics in the metrics module you want to use.
```


## Preprocessing   

**Available dataset names for preprocessing:** human3.6m, amass, 3dpw

Check preprocessing config file: "configs/hydra/preprocess.yaml" for more details.

You can change preprocessor via commandline like below:
```  
mandatory arguments:
  - annotated_data_path  Path of the dataset  
  - dataset             Name of the dataset Ex: 'human3.6m' or '3dpw' (str)  
  - data_type           Type of data to use Ex: 'train', 'validation' or 'test' (str)  
    
optional arguments:  
  - load_60Hz           This value is used only for 3DPW
  - output_name         Name of generated csv file (str) (for default we have specific conventions for each dataset)  
```  
Example:  
```bash
python -m api.preprocess \
    dataset=human3.6m \
    annotated_data_path=$DATASET_PATH \
    data_type=train \
    output_name=new_full \
    data_type=train
```
  
## Training
Check training config file: "configs/hydra/train.yaml" for more details.

You can change training args via command line like below:
```  
mandatory arguments:
  data                  Name of the dataloader yaml file, default is main dataloader (str)
  model                 Name of the model yaml file (str)
  optimizer             Name of the optimizer yaml file, default is adam (str)
  scheduler             Name of the scheduler yaml file, default is reduce_lr_on_plateau (str)
  train_dataset         Path of the train dataset (str)   
  keypoint_dim          Dimension of the data Ex: 2 for 2D and 3 for 3D (int)  
  epochs                Number of training epochs (int) (default: 10)

optional arguments:
  - valid_dataset       Path of validation dataset (str)    
  - normalize           Normalize the data or not (bool)
  - snapshot_interval 	Save snapshot every N epochs (int)
  - load_path           Path to load a model (str) 
  - start_epoch	 	      Start epoch (int)
  - device              Choose either 'cpu' or 'cuda' (str)
  - save_dir            Path to save the model (str)
  - obs_frames_num      Number of observed frames for pose dataset (int)
  - pred_frames_num     Number of future frames for pose dataset (int)
  - model_pose_format   Used data format for pose dataset (str)
  - metric_pose_format  Used data format for metrics if pose dataset is used. If no value is specified it'll use the model_pose_format's value
  - experiment_name:    Experiment name for MLFlow (str) (default: "defautl experiment")
  - mlflow_tracking_uri:  Path for mlruns folder for MLFlow (str) (default: saves mlruns in the current folder)

```  

Example:
```bash  
python -m api.train model=history_repeats_itself \
          train_dataset=$DATASET_TRAIN_PATH \
          valid_dataset=$DATASET_TEST_PATH \
          obs_frames_num=50 \
          pred_frames_num=25
```  

## Evaluation
Check evaluation config file: "configs/hydra/evaluate.yaml" for more details.

You can change evaluation args via command line like below:
``` 
mandatory arguments:
  data          Name of the dataloader yaml file, default is main dataloader (str)
  model         Name of the model yaml file (str)
  dataset       Name of dataset Ex: 'posetrack' or '3dpw' (str)    
  keypoint_dim  Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  load_path     Path to load a model (str)
  save_dir      Path to save output csv file (str)
						   
optional arguments:
  - device      Choose either 'cpu' or 'cuda' (str)
  - obs_frames_num      Number of observed frames for pose dataset (int) (default: 10)
  - pred_frames_num     Number of future frames for pose dataset (int) (default:25)
  - model_pose_format   Used data format for pose dataset (str) (default: xyz)
  - metric_pose_format  Used data format for metrics if pose dataset is used. If no value is specified it'll use the model_pose_format's value
```  

Example:
```bash  
python -m api.evaluate model=msr_gcn \
          dataset=$DATASET_TEST_PATH \
          rounds_num=1 \
          obs_frames_num=10 \
          pred_frames_num=25 \
          load_path=$MODEL_PATH
```  
another example:
```bash
python -m api.evaluate model=zero_vel \
          dataset=$DATASET_TEST_PATH \
          rounds_num=1 \
          obs_frames_num=10 \
          pred_frames_num=25
```

## Generating Outputs

```  
mandatory arguments:
  data			Name of the dataloader yaml file, default is main dataloader (str)
  model			Name of the model yaml file (str)
  dataset    		Name of dataset Ex: 'posetrack' or '3dpw' (str)    
  keypoint_dim          Number of dim data should have Ex: 2 for 2D and 3 for 3D (int)  
  load_path  		Path to load a model (str)  
  pred_frames_num 	Number of frames to predict. Mandatory if load_path is None. (int)
						   
optional arguments:
  save_dir              Path to save the model (str)
  device		Choose either 'cpu' or 'cuda' (str)
```  

Example:
```bash
python -m api.generate_final_output model=st_trans \
          dataset=$DATASET_PATH \
          load_path=$MODEL_CHECKPOINT \
          obs_frames_num=10 \
          pred_frames_num=25 \
          data.is_h36_testing=true \
          save_dir=$OUTPUT_PATH
```  
    
## Visualization  
You can directly change config file: "congifs/hydra/visualize.yaml". Note that you need your model and data: "configs/hydra/data/main.yaml" configs but the default ones should be fine.

Also, all essential changes you need are defined below:
```  
mandatory arguments:  
    dataset_type 	    Name of using dataset. (str)  
    model        	    Name of desired model. (str)  
    images_dir 		    Path to existing images on your local computer (str)
    showing                 Indicates which images we want to show (dash(-) separated list) ([observed, future, predicted, completed])   
    index                   Index of a sequence in dataset to visualize. (int)
    
    
optional arguments:  
  load_path             Path to pretrained model. Mandatory if using a training-based model (str)  
  pred_frames_num 	Number of frames to predict. Mandatory if load_path is None. (int)
```  
  
TODO: add examples