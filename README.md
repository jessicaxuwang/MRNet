# MRNet
Code for MRNet [competition](https://stanfordmlgroup.github.io/competitions/mrnet/).

## Requirement

- [Pytorch](https://pytorch.org/) > 0.4
- [torchvision](https://github.com/pytorch/vision)
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [sklearn](https://scikit-learn.org/stable/)
- Pandas
- Numpy

## Usage

1. Clone this directory
```
git clone https://github.com/jessicaxuwang/MRNet
```

2. Download MRNet dataset from [https://stanfordmlgroup.github.io/competitions/mrnet/]

3. Upzip the file in the `data` directory

4. Edit the `src/train_config.py` 
    * change the `run_dir` to a directory where you want to the check point and log file to be stored
    * change the `data_dir` point to the MRNet data file that was just unzipped
5. Issue the command 
```
python src/train.py src/train_config.py` to train the model
```
