# SpikeBySpike

## How to run

1) Fork this repository and clone it. Afterwards step into the cloned repo.

2) Create a virtual environment running ```python3 -m venv ./venv```

3) Run ```pip install -r requirements.txt``` to install all necessary packages.

4) Run ```smt init SpikeBySpike --executable=spike_by_spike_training.py``` to initialize the sumatra folder.

5) Run the script in standard mode: ```python Start.py parameters.param```

6) After running successfully, run ```smtweb &```. A window in your browser should open and you should be able to view the experiments.
