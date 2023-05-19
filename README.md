# MED

## Install

* Install conda
```
conda env create -f environment.yml
```
```
conda activate MED
```

## Tests
* To run full tests
```
python tests.py --test 1
```
* To see results on graph from one dataset
```
python tests.py --dev_test 1
```
## Visualization
```
python visualizations.py
```
## Running algorithm
* In config folder there are config files used by this project.
* To run one such config file.
```
python main.py --config_file config/numerical.json
```
