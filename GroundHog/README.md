GroundHog by lisa-groundhog
===========================

**WARNING: Groundhog development is over.** Please consider using 
[Blocks](https://github.com/mila-udem/blocks) instead. For an example of machine translation using Blocks please see [Blocks-examples](https://github.com/mila-udem/blocks-examples) repository

GroundHog is a python framework on top of Theano
(http://deeplearning.net/software/theano/) that aims to provide a flexible, yet
efficient way of implementing complex recurrent neural network models. It
supports a variety of recurrent layers, such as DT-RNN, DOT-RNN, RNN with gated
hidden units and LSTM. Furthermore, it enables the flexible combination of
various layers, for instance, to build a neural translation model.

This is a version forked from the original GroundHog
(https://github.com/pascanur/GroundHog) developed by Razvan Pascanu, Caglar
Gulcehre and Kyunghyun Cho. This fork will be the version developed and
maintained by the members of the LISA Lab at the University of Montreal. The
main contributors and maintainers of this fork are currently Dzmitry Bahdanau
and Kyunghyun Cho.

Most of the library documentation is still work in progress, but check the files
containing Tut (in tutorials) for a quick tutorial on how to use the library.

The library is under the 3-clause BSD license, so it may be used for commercial
purposes. 


Installation
------------
To install pytable (important!!)

1. Make sure you have HDF5 version 1.8.4 or above. HDF5 1.10.x is not
supported.

   On OSX you can install HDF5 using `Homebrew <http://brew.sh>`_::

       $ brew tap homebrew/science
       $ brew install hdf5

   On ubuntu::

       $ sudo apt-get install libhdf5-serial-dev

   If you have the HDF5 library in some non-standard location (that
   is, where the compiler and the linker can't find it) you can use
   the environment variable `HDF5_DIR` to specify its location. See
   `the manual
   <http://www.pytables.org/usersguide/installation.html>`_ for more
   details.

4. Install!::

       $ pip install tables




Neural Machine Translation (path = experiment/nmt')
--------------------------

 1. Prepare corpus: Run pre-processing with parallel corpus (E.g. from wang2vec)
 2. Config model Paramters: Set path and parameters in state.py (e.g. update prototype_phrase_state in state.py)
 3. Build model: Run train.py (e.g. python train.py --proto=prototype_phrase_state)
 4. test: run sample.py (elipse need to set to theano_flag in eclipse: http://computingstar.blogspot.hk/2012/12/theano-gpu-setting.html)

