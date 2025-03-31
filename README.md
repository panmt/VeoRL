## Setting up

#### Meta-World
```
conda env create -f set_up/metaworld.yaml
```

#### CARLA

1. Create an environment
```
conda env create -f set_up/carla.yaml
```

2. Download and setup CARLA 0.9.10
```
chmod +x set_up/setup_carla.sh
./setup_carla.sh
```

3. Add to your python path:
```
export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI
export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```
4. merge the directories, i.e., put 'carla_env_dream.py' into 'CARLA_0.9.10/PythonAPI/carla/agents/navigation/'.

#### MineDojo

1. Create an environment
```
conda create -n minedojo python=3.9
conda activate minedojo 
```

2. Install Java: JDK `1.8.0_171`. Then install the [MineDojo](https://github.com/MineDojo/MineDojo) environment and [MineCLIP](https://github.com/MineDojo/MineCLIP) following their official documents. 

3. Install dependencies
    ```
    pip install -r set_up/requirements.txt
    ```

4. Download the MineCLIP weight [here](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view?usp=sharing) and place them at `./weights/mineclip_attn.pth`.

## Training

#### Meta-World:

```
python dreamer.py --logdir path/to/log --config defaults metaworld --task metaworld_handle_press  --target_dataset_logdir path/to/offline_dataset --source_video_logdir path/to/video 
```

#### CARLA

Terminal 1:
```
cd CARLA_0.9.10
bash CarlaUE4.sh -fps 20 -opengl
```

Terminal 2:
```
python dreamer.py --logdir path/to/log --config defaults carla  --target_dataset_logdir path/to/offline_dataset --source_video_logdir path/to/video 
```

When running other environments where the CARLA simulator is not deployed, it may be necessary to comment out the line 'from agents.navigation.carla_env_dreamer import CarlaEnv' in the dreamer.py file.

#### MineDojo

```
python dreamer.py --logdir path/to/log --config defaults minedojo --task minedojo_dv2_harvest_log_in_plains  --target_dataset_logdir path/to/offline_dataset --source_video_logdir path/to/video 
```
