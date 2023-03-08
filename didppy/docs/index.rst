DIDPPy
======

DIDPPy is a Python interface to use Domain-Independent Dynamic Programming (DIDP), which is a model-based framework to solve combinatorial optimization problems such as vehicle routing problems (VRPs) and scheduling problems.
In this framework, once you define a problem as a DP model, you can use generic solvers to solve it.

Installation
------------
You can install ``didppy`` from PyPI using ``pip``.
You need to use Python 3.7 or higher.

.. code-block:: bash

   pip install didppy

.. toctree::
    :maxdepth: 1
    :caption: Introduction

    quickstart
    tutorial
    papers

.. toctree::
    :maxdepth: 1
    :caption: Examples

    examples/cvrp
    examples/salbp-1
    examples/mosp
    examples/talent-scheduling

.. toctree::
    :maxdepth: 1
    :caption: User Guide

    debugging
    solver-selection

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    
    reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
