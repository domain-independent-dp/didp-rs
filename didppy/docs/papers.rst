DIDP Papers
===========

*  Ryo Kuroiwa and J. Christopher Beck. `Domain-Independent Dynamic Programming: Generic State Space Search for Combinatorial Optimization <https://ojs.aaai.org/index.php/ICAPS/article/view/27200/26973>`_. *In Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS).* 2023.

    * This paper introduces the paradigm of DIDP, :class:`~didppy.CAASDy`, and DIDP models for :ref:`TSPTW <tutorial:TSPTW>`, the capacitated vehicle routing problem (CVRP), bin packing, the simple assembly line balancing problem (SALBP-1), :ref:`MOSP <advanced-tutorials/general-cost:MOSP>`, and :ref:`graph-clear <advanced-tutorials/transition-dominance:Graph-Clear>`.

* Ryo Kuroiwa and J. Christopher Beck. `Solving Domain-Independent Dynamic Programming Problems with Anytime Heuristic Search <https://ojs.aaai.org/index.php/ICAPS/article/view/27201/26974>`_. *In Proceedings of the 33rd International Conference on Automated Planning and Scheduling (ICAPS).* 2023.

    * This paper introduces anytime solvers for DIDP including :class:`~didppy.CABS` and DIDP models for the multi-commodity pick-and-delivery traveling salesperson problem (m-PDTSP), the :ref:`talent scheduling problem <advanced-tutorials/forced-transitions:Talent Scheduling>`, and the single machine scheduling to minimize total weighted tardiness (:math:`1||\sum w_iT_i`).

* Ryo Kuroiwa and J. Christopher Beck. `Large Neighborhood Beam Search for Domain-Independent Dynamic Programming <https://drops.dagstuhl.de/storage/00lipics/lipics-vol280-cp2023/LIPIcs.CP.2023.23/LIPIcs.CP.2023.23.pdf>`_. *In Proceedings of the 29th International Conference on Principles and Practice of Constraint Programming (CP).* 2023.

    * This paper introduces Large Neighborhood Beam Search (LNBS).

* Ryo Kuroiwa and J. Christopher Beck. `Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming <https://tidel.mie.utoronto.ca/pubs/aaai24-parallel-camera-ready.pdf>`_. *In Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence (AAAI).* 2024.

    * This paper parallelizes :class:`~didppy.CABS`.

* Ryo Kuroiwa and J. Christopher Beck. `Domain-Independent Dynamic Programming <https://arxiv.org/pdf/2401.13883.pdf>`_. *arXiv*. 2024.

    * This paper provides formal definitions of the modeling language and solvers for DIDP. It also introduces DIDP models for the orienteering problem with time windows and the multi-dimensional knapsack problem.

* J. Christopher Beck, Ryo Kuroiwa, Jimmy H.M. Lee, Peter J. Stuckey, and Allen Z. Zhong. `Transition Dominance in Domain-Independent Dynamic Programming <https://tidel.mie.utoronto.ca/pubs/Transition_Dominance_in_DIDP.pdf>`_. *In Proceedings of the 31st International Conference on Principles and Practice of Constraint Programming (CP).* 

    * This paper introduces state functions and :doc:`transition dominance <advanced-tutorials/transition-dominance>`.

