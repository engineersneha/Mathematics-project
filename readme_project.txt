Requirements:

# Install Anaconda and set up conda environment (for windows)

# Update package manager 

# Update conda 

conda update --all 

conda update anaconda 

# Install libraries
numpy 
pysindy
matplotlib.pyplot
scipy
xarray
psutil
pandas
pickle
pathlib
time
sys

PySINDy code for analysis of Lorenz system: PySINDy_lorenz.py
- creates sindy model based on training data generated and identifies Lorenz attractor from data
- Model tested with new set of test data
PySINDy code for analysis of Rossler system: PySINDy_rossler.py
- creates sindy model based on training data generated and identifies Rossler attractor from data
- Model tested with new set of test data
PySINDy code for SPOD test (mode extraction): spod_test.py
- Runs SPOD to obtain modes at a certain frequency 
- Passes these modes as numpy array to sindy model 
