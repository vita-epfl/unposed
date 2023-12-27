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


# Results
Here you can find their results on the Human3.6M dataset:

| **Model**                  | **$80 ms$** | **$160 ms$** | **$320 ms$** | **$400 ms$** | **$560 ms$** | **$720 ms$** | **$880 ms$** | **$1000 ms$** |
|----------------------------|-------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|
| **STS-GCN**                | 17.7        | 33.9         | 56.3         | 67.5         | 85.1         | 99.4         | 109.9        | 117.0         |
| **STS-GCN + pUAL (ours)**  | 13.2        | 27.1         | 54.7         | 66.2         | 84.5         | 97.9         | 109.3        | 115.7         |
| **HRI\***                   | 12.7        | 26.1         | 51.5         | 62.6         | 80.8         | 95.1         | 106.8        | 113.8         |
| **HRI\* + pUAL (ours)**     | 11.6        | 25.3         | 51.2         | 62.2         | 80.1         | 93.7         | 105.0        | 112.1         |
| **PGBIG**                  | 10.3        | 22.6         | 46.6         | 57.5         | 76.3         | 90.9         | 102.7        | 110.0         |
| **PGBIG + pUAL (ours)**    | 9.6         | 21.7         | 46.0         | 57.1         | 75.9         | 90.3         | 102.1        | 109.5         |
| **ST-Trans**               | 13.0        | 27.0         | 52.6         | 63.2         | 80.3         | 93.6         | 104.7        | 111.6         |
| **ST-Trans + pUAL (ours)** | 10.4        | 23.4         | 48.4         | 59.2         | 77.0         | 90.7         | 101.9        | 109.3         |


Similarly on the AMASS dataset:

| **Model**                  | **$80 ms$** | **$160 ms$** | **$320 ms$** | **$400 ms$** | **$560 ms$** | **$720 ms$** | **$880 ms$** | **$1000 ms$** |
|-------------------------|------|-------|-------|-------|-------|-------|-------|--------|
| **STS-GCN**                 | 13.9  | 27.6  | 32.0  | 43.1  | 51.2  | 59.2  | 63.9  | 68.7   |
| **STS-GCN + pUAL**          | 13.0  | 27.0  | 31.6  | 42.4  | 50.6  | 59.1  | 63.5  | 68.1   |
| **HRI**                     | 13.5  | 27.0  | 31.3  | 42.0  | 50.3  | 58.6  | 63.1  | 67.2   |
| **HRI + pUAL**              | 12.8  | 25.2  | 31.1  | 41.4  | 49.8  | 58.1  | 62.7  | 66.5   |
| **PGBIG**                   | 14.1  | 28.4  | 32.7  | 43.6  | 51.8  | 59.9  | 64.6  | 67.9   |
| **PGBIG + pUAL**            | 13.2  | 26.5  | 32.3  | 40.9  | 49.5  | 58.1  | 64.4  | 66.9   |
| **ST-Trans**                | 13.6  | 27.3  | 31.9  | 42.5  | 50.4  | 58.3  | 64.0  | 66.6   |
| **ST-Trans + pUAL**         | 12.1  | 24.8  | 30.8  | 39.7  | 47.8  | 56.5  | 63.8  | 66.7   |

and the 3DPW dataset:
| **Model**                  | **$80 ms$** | **$160 ms$** | **$320 ms$** | **$400 ms$** | **$560 ms$** | **$720 ms$** | **$880 ms$** | **$1000 ms$** |
|-------------------------|------|-------|-------|-------|-------|-------|-------|--------|
| **STS-GCN**                 | 13.5 | 26.2  | 31.4  | 40.3  | 47.7  | 55.0  | 60.0  | 62.4   |
| **STS-GCN + pUAL**          | 12.8 | 25.9  | 31.2  | 40.0  | 47.3  | 54.8  | 59.8  | 62.2   |
| **HRI**                     | 15.9 | 30.5  | 33.8  | 45.0  | 53.5  | 62.9  | 67.6  | 72.5   |
| **HRI + pUAL**              | 14.8 | 29.6  | 33.2  | 44.6  | 53.2  | 62.4  | 67.0  | 72.2   |
| **PGBIG**                   | 13.1 | 25.5  | 37.0  | 48.8  | 57.8  | 66.9  | 71.6  | 75.0   |
| **PGBIG + pUAL**            | 12.2 | 23.5  | 36.0  | 47.1  | 55.7  | 66.4  | 71.4  | 74.5   |
| **ST-Trans**                | 12.1 | 24.5  | 37.0  | 47.4  | 57.6  | 64.6  | 70.6  | 73.8   |
| **ST-Trans + pUAL**         | 11.1 | 22.3  | 35.0  | 45.7  | 53.6  | 63.6  | 70.0  | 73.2   |