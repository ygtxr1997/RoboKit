# RoboKit

## Install

Enter your Python environment (we use Python 3.10), and then:

```shell
git clone https://github.com/ygtxr1997/RoboKit.git
cd RoboKit
pip install -e .
```

## Data Format

Download and unzip the data file (e.g. `collected_data_0507.zip`) to your data root 
`YOUR_PATH_TO_DATA_ROOT/`:

### 1. Preprocess the data

```shell
python scripts/01_preprocess_data.py -R "YOUR_PATH_TO_DATA_ROOT/"
```

The saved `statistics.json` file would be helpful for action norm and unnorm.
Then you can run our provided script to validate the data format:

```shell
python tests/test_data_format.py -R "YOUR_PATH_TO_DATA_ROOT/"
```

The output could be:
```shell
737-th data:: Dict,keys=dict_keys(['primary_rgb', 'gripper_rgb', 'primary_depth', 'gripper_depth', 'language_text', 'actions', 'rel_actions', 'robot_obs'])
primary_rgb,<class 'numpy.ndarray'>,shape=(480, 848, 3)
gripper_rgb,<class 'numpy.ndarray'>,shape=(480, 848, 3)
primary_depth,<class 'numpy.ndarray'>,shape=(480, 848)
gripper_depth,<class 'numpy.ndarray'>,shape=(480, 848)
language_text:<class 'str'>,len=18
actions,<class 'numpy.ndarray'>,shape=(7,)
rel_actions,<class 'numpy.ndarray'>,shape=(7,)
robot_obs,<class 'numpy.ndarray'>,shape=(14,)
```

About `actions` and `rel_actions` (they are same for now):
```shell
actions: (x,y,z,row,pitch,yaw,g)
rel_actions: (j_x,j_y,j_z,j_ax,j_ay,j_az,g)
```

About `robot_obs`:
```shell
(tcp pos (3), tcp ori (3), gripper width (1), joint_states (6) in rad, gripper_action (1)
```

### 2. TCLDataset usage

```shell
# In your .py file, to use TCLDataset provided by robokit
from robokit.data.tcl_datasets import TCLDataset

my_tcl_dataset = TCLDataset(
  root="YOUR_PATH_TO_DATA_ROOT/",  # data root
  use_extracted=True,  # load `rel_actions` from a single npy file rather than load several npz files
  load_keys=["rel_actions", "primary_rgb", "gripper_rgb", "robot_obs", "language_text"],
  # default (None) will load all keys: ["primary_rgb", "gripper_rgb", "primary_depth", "gripper_depth", "language_text", "actions", "rel_actions", "robot_obs"]
  # you can modify it as shown above
)
```
