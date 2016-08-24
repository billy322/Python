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

3. For stability (and performance too) reasons, it is strongly
   recommended that you install the C-Blosc library separately,
   although you might want PyTables to use its internal C-Blosc
   sources.

3. Optionally, consider to install the LZO compression library and/or
   the bzip2 compression library.

4. Install!::

       $ pip install tables

5. To run the test suite run::

       $ python -m tables.tests.test_all

   If there is some test that does not pass, please send the
   complete output for tests back to us.

To install Groundhog in a multi-user setting (such as the LISA lab)

``python setup.py develop --user``

For general installation, simply use

``python setup.py develop``

NOTE: This will install the development version of Theano, if Theano is not
currently installed.

Neural Machine Translation
--------------------------

See experiments/nmt/README.md

