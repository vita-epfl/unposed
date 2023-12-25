Using the commands below you can train different models on different datasets. 

**NOTE**: AMASS and 3DPW settings are simillar to each other.

## ST_Trans
### Human3.6M
```bash
python -m api.train model=st_trans \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=50 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=human3.6m \
    model.n_major_joints=22 \
    model.loss.nJ=32\
    epochs=15
```
### AMASS
```bash
python -m api.train model=st_trans \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=AMASS \
    model.n_major_joints=18 \
    model.loss.nJ=18 
```
### 3DPW
```bash
python -m api.train model=st_trans \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.loss.nT=25 \
    model.pre_post_process=3DPW \
    model.n_major_joints=18 \
    model.loss.nJ=18
```
## PGBIG
### Human3.6M
```bash
python -m api.train model=pgbig \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=human3.6m \
    model.in_features=66 \
    model.loss.nJ=22 \
    model.loss.pre_post_process=human3.6m \
    epochs=50
```
### AMASS
```bash
python -m api.train model=pgbig \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=AMASS \
    model.in_features=54 \
    model.loss.nJ=18 \
    model.loss.pre_post_process=AMASS \
    epochs=50
```
### 3DPW
```bash
python -m api.train model=pgbig \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=3DPW \
    model.in_features=54 \
    model.loss.nJ=18 \
    model.loss.pre_post_process=3DPW \
    epochs = 50
```
## History-Repeats-Itself
### Human3.6M
```bash
python -m api.train model=history_repeats_itself \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    model.modality=Human36 \
    model.in_features=66 \
    obs_frames_num=50 \
    pred_frames_num=25
```
### AMASS
```bash
python -m api.train model=history_repeats_itself \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    model.modality=AMASS \
    model.in_features=66 \
    obs_frames_num=50 \
    pred_frames_num=25
```

### 3DPW
```bash
python -m api.train model=history_repeats_itself \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    model.modality=3DPW \
    model.in_features=66 \
    obs_frames_num=50 \
    pred_frames_num=25
```

## STS-GCN
### Human3.6M
```bash
python -m api.train model=sts_gcn \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=human3.6m \
    model.n_major_joints=22 \
    model.loss.nJ=32
```
### AMASS
```bash
python -m api.train model=sts_gcn \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=AMASS \
    model.n_major_joints=18 \
    model.loss.nJ=18
```

### 3DPW
```bash
python -m api.train model=sts_gcn \
    train_dataset=$DATASET_TRAIN_PATH \
    valid_dataset=$DATASET_VALIDATION_PATH \
    obs_frames_num=10 \
    pred_frames_num=25 \
    model.pre_post_process=3DPW \
    model.n_major_joints=18 \
    model.loss.nJ=18
```