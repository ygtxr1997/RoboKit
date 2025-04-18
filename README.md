# RoboKit

## Install

Enter your Python environment (we use Python 3.10), and then:

```shell
git clone https://github.com/ygtxr1997/RoboKit.git
cd RoboKit
pip install -e .
```

## Data Format

Assuming your data root dir is `YOUR_PATH_TO_DATA_ROOT/`.

Then you can run our provided script to validate the data format:

```shell
python tests/test_data_format.py -R YOUR_PATH_TO_DATA_ROOT/
```

