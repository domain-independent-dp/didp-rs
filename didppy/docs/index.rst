.. DIDPPy documentation master file, created by
   sphinx-quickstart on Fri Jul 22 13:13:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DIDPPy API Reference
==================================

Modeling
========

Model
-----
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.Model
   didppy.Transition

Expressions
-----------
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.ElementExpr
   didppy.SetExpr
   didppy.IntExpr
   didppy.FloatExpr
   didppy.Condition

Functions
---------
.. autosummary::
   :toctree: _autosummary

   didppy.max
   didppy.min
   didppy.sqrt
   didppy.log
   didppy.float

State
-----
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.State

Variables
---------
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.ElementVar
   didppy.ElementResourceVar
   didppy.SetVar
   didppy.IntVar
   didppy.IntResourceVar
   didppy.FloatVar
   didppy.FloatResourceVar

Tables of Constants
-------------------
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.ElementTable1D
   didppy.ElementTable2D
   didppy.ElementTable3D
   didppy.ElementTable
   didppy.SetTable1D
   didppy.SetTable2D
   didppy.SetTable3D
   didppy.SetTable
   didppy.IntTable1D
   didppy.IntTable2D
   didppy.IntTable3D
   didppy.IntTable
   didppy.FloatTable1D
   didppy.FloatTable2D
   didppy.FloatTable3D
   didppy.FloatTable
   didppy.BoolTable1D
   didppy.BoolTable2D
   didppy.BoolTable3D
   didppy.BoolTable

Others
------
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.ObjectType
   didppy.SetConst

Solving
=======

Solvers
-------
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.ForwardRecursion
   didppy.CABS
   didppy.CAASDy
   didppy.DFBB
   didppy.CBFS
   didppy.ACPS
   didppy.APPS
   didppy.DBDFS
   didppy.BreadthFirstSearch
   didppy.WeightedAstar
   didppy.Dijkstra

Solution
--------
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.Solution

Solver Configurations
---------------------
.. autosummary::
   :nosignatures:
   :toctree: _autosummary

   didppy.FOperator


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
