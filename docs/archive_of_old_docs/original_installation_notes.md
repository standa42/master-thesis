conda create -n myenv python=3.6
conda activate myenv

conda install -c conda-forge scikit-learn 

maybe consider https://stackoverflow.com/questions/65273118/why-is-tensorflow-not-recognizing-my-gpu-after-conda-install/65709577#65709577
conda install -c anaconda tensorflow-gpu

python -m ipykernel install --user --name myenv --display-name "Python (myenv)"

conda install jupyter ipykernel

jupyter notebook

python -m ipykernel install --user --name myenv --display-name "Python (myenv)"

jupyter notebook

I have installed in user folder (Standa) and then switched to desired folder only on first 'jupyter notebook call'

conda install keras jupyter ipykernel numpy pandas matplotlib seaborn pillow opencv 




// how to freeze conda env
https://stackoverflow.com/questions/41249401/difference-between-pip-freeze-and-conda-list
conda list --export > requirements.txt
conda create --name <envname> --file requirements.txt

// theoretical approach to create pip env from conda env
https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

python was 3.8? (or maybe 3.7)


env12 has python 3.7 and gpu from that github answer

added kivy from conda install kivy -c conda-forge

freeze env conda env export > environment-tfg7.yml