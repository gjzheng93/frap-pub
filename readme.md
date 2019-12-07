# Reinforcement learning for traffic signal control

This is a reinforcement learning pipeline for intelligent traffic signal control. Usage and more information can be found below.

## Installation

We recommend to run the code through docker. Some brief documentation can be found at https://docs.docker.com/.

1. Please pull the docker image from the docker hub. 
``docker pull gjzheng/frap:latest``

2. Please run the built docker image to initiate a docker container. Please remember to mount the code directory.

```
docker run -it -v path/to/the/code/repo/:/work/ simulator-test bash
cd /work/
bash example_run.sh
```

## Usage

Start an example:

``sh example_run.sh``

In this script, it runs the file ``run_batch.py``. Here are some important arguments that can be modified for different experiments:

* memo: the memo name of the experiment
* algorithm: the specified algorithm, e.g., TransferDQN.
* num_phase: phase setting for the experiment, e.g., 2, 4, or 8. Default is 8. Other settings may need extra data.

Hyperparameters such as learning rate, sample size and the like for the agent can also be assigned in our code and they are easy to tune for better performance.

## Agent

* ``agent.py``

  A abstract class of different agents.

* ``network_agent.py``

  A abstract class of neural network based agents.  All methods are defined in this file but ``build_network()``, which means only ``build_network()`` is necessary in specific network agents.

* ``transfer_dqn_agent.py``

  This is our proposed framework which can handle all-phase scenarios and achieve invariance to symmetrical transformation like flipping and rotation.

## Others

More details about this project are demonstrated in this part.

* ``config.py`` 

  The whole configuration of this project. Note that some parameters will be replaced in ``runexp.py`` while others can only be changed in this file, please be very careful!!!

* ``pipeline.py``

  The whole pipeline is implemented in this module:

  Start a SUMO environment, run a simulation for certain time(one round), construct samples from raw log data, update the model and model pooling.

* ``generator.py``

  A generator to load a model, start a SUMO enviroment, conduct a simulation and log the results.

* ``sumo_env.py``

  Define a SUMO environment to interact with SUMO and obtain needed data like features.

* ``anon_env.py``

  Implement a multi-process version of SUMO.

* ``construct_sample.py``

  Construct training samples from original data. Select desired state features in the config and compute the corrsponding average/instant reward with specific measure time.

* ``updater.py``

  Define a class of updater for model updating.
