# SpikeBySpike

## How to run

1) Clone this repository. Afterwards step into the cloned repo.

2) Create a virtual environment using ```$ python3 -m venv /venv```

3) Activate the virtual environment using ```$ source venv/bin/activate```

4) Download all necessary packages using ```$ pip3 install -r requirements.txt```

5) (Without Sumatra) Start the main program using ```$ python3 main.py Data parameters.param```
This will run the learning process and store important files in a local ```Resources``` folder. You can then open ```main.py``` and change ```TRAINING``` to ```False``` and execute the script again. This will produce plots for testing.

6) (With Sumatra) Sumatra makes a new commit to the repository. If you are not authorized to commit to this repository, please fork this repository. Execute ```$ smt init SpikeBySpike```.

7) After the ```.smt``` folder was created you can start an experiment by executing ```python3 Start.py parameters.param```.
