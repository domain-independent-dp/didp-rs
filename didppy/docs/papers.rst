DIDP Papers
===========

*  Ryo Kuroiwa and J. Christopher Beck. `Domain-Independent Dynamic Programming: Generic State Space Search for Combinatorial Optimization <https://doi.org/10.1609/icaps.v33i1.27200>`_. *In Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS).* 2023.

    * This paper introduces the paradigm of DIDP, :class:`~didppy.CAASDy`, and DIDP models for :ref:`TSPTW <tutorial:TSPTW>`, the capacitated vehicle routing problem (CVRP), bin packing, the simple assembly line balancing problem (SALBP-1), :ref:`MOSP <advanced-tutorials/general-cost:MOSP>`, and :ref:`graph-clear <advanced-tutorials/transition-dominance:Graph-Clear>`.

* Ryo Kuroiwa and J. Christopher Beck. `Solving Domain-Independent Dynamic Programming Problems with Anytime Heuristic Search <https://doi.org/10.1609/icaps.v33i1.27201>`_. *In Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS).* 2023.

    * This paper introduces anytime solvers for DIDP including :class:`~didppy.CABS` and DIDP models for the multi-commodity pick-and-delivery traveling salesperson problem (m-PDTSP), the :ref:`talent scheduling problem <advanced-tutorials/forced-transitions:Talent Scheduling>`, and the single machine scheduling to minimize total weighted tardiness (:math:`1||\sum w_iT_i`).

* Ryo Kuroiwa and J. Christopher Beck. `Large Neighborhood Beam Search for Domain-Independent Dynamic Programming <https://doi.org/10.4230/LIPIcs.CP.2023.23>`_. *In Proceedings of the 29th International Conference on Principles and Practice of Constraint Programming (CP).* 2023.

    * This paper introduces Large Neighborhood Beam Search (LNBS).

* Ryo Kuroiwa and J. Christopher Beck. `Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming <https://doi.org/10.1609/aaai.v38i18.30062>`_. *In Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI).* 2024.

    * This paper parallelizes :class:`~didppy.CABS`.

* Ryo Kuroiwa and J. Christopher Beck. `Domain-Independent Dynamic Programming <https://doi.org/10.1016/j.artint.2026.104506>`_. *Artificial Intelligence*. 2026.

    * This paper provides formal definitions of the modeling language and solvers for DIDP. It also introduces DIDP models for the orienteering problem with time windows and the multi-dimensional knapsack problem.

* J. Christopher Beck, Ryo Kuroiwa, Jimmy H.M. Lee, Peter J. Stuckey, and Allen Z. Zhong. `Transition Dominance in Domain-Independent Dynamic Programming <https://doi.org/10.4230/LIPIcs.CP.2025.5>`_. *In Proceedings of the 31st International Conference on Principles and Practice of Constraint Programming (CP).* 

    * This paper introduces state functions and :doc:`transition dominance <advanced-tutorials/transition-dominance>`.

* Ryo Kuroiwa and Edward Lam. `Column Generation with Domain-Independent Dynamic Programming <https://doi.org/10.4230/LIPIcs.CP.2026.37>`_. *In Proceedings of the 32nd International Conference on Principles and Practice of Constraint Programming (CP).* 2026.

    * This paper combines column generation with DIDP, introducing :class:`~didppy.SetResourceVar`, :doc:`higher-order expressions <advanced-tutorials/higher-order-expression>`, and :func:`~didppy.fractional_knapsack`.