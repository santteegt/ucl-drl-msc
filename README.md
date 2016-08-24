# University College London - MSc Thesis Project #

# Collaborative Filtering using a Deep Reinforcement Learning Approach

******

Author
======

* [Santiago Gonzalez Toral](hernan.toral.15@ucl.ac.uk) | MSc WSBDA Candidate

Supervisors
======

* [PhD. Jun Wang]() | MSc WSBDA Director & Senior Lecturer


Overview
======

.. contents:: **Contents of this document**
   :depth: 2

System Requirements and Setup
======

- `Python 2.7`
- `Virtualenv`
- `Tensorflow`
- `Jupyter notebook`


Dependencies
======

Installation
======

```bash
$ git clone https://santteegt@bitbucket.org/msc_drl/ucl-cfdrl-msc.git

$ cd ucl-cfdrl-msc
$ git submodule update --init --recursive
$ mkdir .venv
$ virtualenv --system-site-packages --python=python2.7 .venv/
$ source .venv/bin/activate

(venv)$ cd gym
(venv)$ pip install -e .
# installation for Mac OS X. For other platforms, refer to https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#virtualenv-installation
(venv)$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl
(venv)$ pip install --upgrade $TF_BINARY_URL
(venv)$ pip install pymongo pandas gensim fastFM matplotlib
```
## FLANN Installation

Download Flann from (Official Site)[http://www.cs.ubc.ca/research/flann/] and extract the contents to the project home directory

Pre-requisite: (Cmake)[https://cmake.org/]

```bash
(venv)$ cd flann-1.8.4-src
(venv)$ mkdir build
(venv)$ cd build
(venv)$ cmake ..
(venv)$ execute cmake-gui and set PYTHONPATH=${project_dir}/.venv/lib/python2.7
(venv)$ make
(venv)$ sudo make install
```

Running the environment
======

```bash
(venv)$ mkdir ddpg-results
(venv)$ python src/run.py --outdir ddpg-results/experiment1 --env CollaborativeFiltering-v0
```

License and Version
======