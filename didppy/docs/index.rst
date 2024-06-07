DIDPPy
======

DIDPPy is a Python interface to use Domain-Independent Dynamic Programming (DIDP).
DIDP is a model-based framework for combinatorial optimization problems such as vehicle routing problems (VRPs) and scheduling problems.
With DIDP, we can use dynamic programming (DP) to solve combinatorial optimization problems without implementing DP algorithms.
Just like in mixed-integer programming (MIP), once we define a problem as a model, we can use a solver provided by the framework to solve the model.

Installation
------------
``didppy`` can be installed from PyPI using ``pip``.
Python 3.7 or higher is required.

.. code-block:: bash

   pip install didppy

.. toctree::
    :maxdepth: 1
    :caption: Introduction

    quickstart
    tutorial
    advanced-tutorials
    examples
    papers
    references

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    solver-selection
    dump-and-load
    debugging

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    
    api-reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
