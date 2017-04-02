## Learning to Recommend using a Deep Reinforcement Agent

This is a project developed as part of my thesis dissertation for the M.Sc programme in Web Science and Big Data Analytics at University College London.

It implements a Deep Reinforcement Learning algorithm that learns to give good recommendations to users under a simulated environment built using the OpenAI gym framework. For more information, refer to the main [document](https://github.com/santteegt/ucl-drl-msc/blob/master/latex_doc/Main.pdf).
******

### Author
* [Santiago Gonzalez Toral](hernan.toral.15@ucl.ac.uk) | M.Sc Web Science and Big Data Analytics (WSBDA) candidate

### Supervisor

* [Dr. Jun Wang]() | Reader at University College London. Programme Director of M.Sc WSBDA

### Overview

1. [Project-Dependencies](#project-dependencies)
2. [Setup](#setup)
3. [Datasets](#datasets)
4. [Execution](#execution)
5. [License](#license)

---

### Project-Dependencies

The following packages are needed to execute the project models

- `Python 2.7`
- `Virtualenv`
- `OpenAI gym` (included in this project)
- `Tensorflow`
- `Jupyter notebook`
- `Graphlab Create API 2.1` (with Academic License)


### Setup

To setup the environment, the following commands need to be executed:

```bash
$ git clone https://github.com/santteegt/ucl-drl-msc.git

$ cd ucl-drl-msc
$ git submodule update --init --recursive
$ mkdir .venv
$ virtualenv --system-site-packages --python=python2.7 .venv/
$ source .venv/bin/activate

(venv)$ cd gym
(venv)$ pip install -e .
# installation for Mac OS X. For other platforms, refer to https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#virtualenv-installation
(venv)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl
(venv)$ pip install --upgrade $TF_BINARY_URL
(venv)$ pip install pymongo pandas gensim fastFM matplotlib scikit-learn scipy
(venv)$ pip install --upgrade --no-cache-dir https://get.graphlab.com/GraphLab-Create/2.1/hernan.toral.15@ucl.ac.uk/55E9-2088-3AF8-120F-D171-1AAB-95A3-B077/GraphLab-Create-License.tar.gz

```

### Datasets

[Movielens](http://grouplens.org/datasets/movielens/) datasets files were pre-processed for running the models. They can be downloaded from the following URL: [Dataset files](https://mega.nz/#F!oFYFmZyQ!UERNs4e3Jvvkml2zNj2ngA) [Alternative 2](https://drive.google.com/open?id=0BxTIDC-8xn61bDRyT25SM1k5MTg)

Then, 

* Extract and copy the contents of FMmodel.zip to the `data` folder
* To use the Movielens-100k dataset, extract the contents of the data-movielens100k.zip file into the `data` folder
* To use the Movielens-1M dataset, extract the contents of the data-movielens1m.zip file into the `data` folder

### Execution

To make the running process more easier, executable files for each experiment are provided under the `bin` folder (make sure to activate the virtualenv before running the exec file):

1. **DRL-kNN-CB**: `run-v2.sh` runs the model for the Movielens 100k dataset. `run-v0.sh` runs the model for the Movielens 1M dataset
2. **DRL-kNN-CF**: `run-v1.sh` runs the model for the Movielens 100k dataset.
3. **DRL-FM**: `run-v3.sh` runs the model for the Movielens 100k dataset.
4. **Random agent**: `run-v1-random.sh` `run-v2-random.sh` runs the DDPG algorithm using a random agent

Human readable rendering of recommendations made while running the model can be seen using the following command:

```bash
(venv)$ cd ucl-drl-msc
(venv)$ tail -F run.log
```

Reward logs obtained by an agent while running the model can be read using the following command:

```bash
(venv)$ cd ucl-drl-msc
(venv)$ tail -F ddpg-results/experiment1/output.log
```
Additionally, to visualize the model parameters during training, you can run the Tensorflow dashboard using the following command:

```bash
(venv)$ cd ucl-drl-msc
(venv)$ python src/dashboard.py --exdir ddpg-results/
```

Finally, to stop the training process, it is necessary to run the following command:
```bash
(venv)$ cd ucl-drl-msc/bin
(venv)$ sh stop.sh
```

### License

The Project code is mostly developed under the Apache2 license, except for the module developed using Graphlab Create which is based on an Academic License provided by Turi
