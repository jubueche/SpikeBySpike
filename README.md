# SpikeBySpike

## How to run

1) Clone this repository. Afterwards step into the cloned repo.

2) Create a virtual environment using ```$ python3 -m venv venv/```

3) Activate the virtual environment using ```$ source venv/bin/activate```

4) Download all necessary packages using ```$ pip3 install -r requirements.txt```

5) Simulation: Execute ```$ python main.py -h``` to see the command line options.

6) DYNAPS: Navigate to the DYANAPS subdirectory and execute ```python main.py -h``` to see available commands.

Note: In order to run the DYNAPS in the loop, plugin in the DYNAPS, start cortexcontrol and start the
rpyc server in headless mode.
