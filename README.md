# master-thesis-implementation
Implementation part of my master thesis.

## Requirements

Main project development is currently done in conda environment (under python 3.7), because it enables quick start with tensorflow-gpu. Build for other environment like pipenv will be probably added in the future.

## How to set up project

1. Clone repository and change current directory
```
git clone https://github.com/standa42/master-thesis-implementation.git
cd "master-thesis-implementation"
```

2. Install environment
```
cd rims

# create conda environment (current name is tfg7, contains tensorflow-gpu)
conda env create -f environment-tfg7.yml
conda activate tfg7

# enable referencing in thesis package
pip install -e "."
```

3. Add environment to jupyter notebooks (optional)
```
python -m ipykernel install --user --name tfg7 --display-name "Python Conda (tfg7)"
```

## Project structure
- bin - standalone scripts mostly doing one step of the processing pipeline
- config - configurations
- data - folder containing all processed data
- docs - documentation
- notebooks - jupyter notebooks
- src - main source folder containing important modules








