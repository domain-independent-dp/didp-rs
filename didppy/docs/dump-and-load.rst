Dump and Load Model
===================

DIDPPy allows a user to extract a :class:`~didppy.Model` object as strings or YAML files. 

Dump and Load String Representation
-----------------------------------

Users can extract the string representation of a :class:`~didppy.Model` object with the method :meth:`~didppy.Model.dump_to_str`. 
The method returns two strings: a domain string and a problem string, and they can be loaded back into a :class:`~didppy.Model` object with the method :meth:`~didppy.Model.load_from_str`.

Example:

.. code-block:: python

    import didppy as dp
    model = dp.Model()
    v = model.add_int_var(target=0)
    model.add_transition(dp.Transition(name="increase", effects=[(v, v+1)]))
    model.add_base_case([v >= 10])
    domain, problem = model.dump_to_str()
    reloaded_model = dp.Model.load_from_str(domain, problem)

The above example shows the dump and load process of a simple model. To successfully load a meaningful model,
the domain string and the problem string must contain information of state variables, transitions, and base cases.
Both the domain string and the problem string have the `YAML syntax <https://spacelift.io/blog/yaml>`_, and they can be directly written into YAML files.

Dump and Load YAML Files
------------------------

If the user wants to store the string representation of a :class:`~didppy.Model` object as files, it can be achieved by calling :meth:`~didppy.Model.dump_to_files` directly. 
Similarly, the files can be loaded back into a :class:`~didppy.Model` object with the method :meth:`~didppy.Model.load_from_files`.

Example:

.. code-block:: python

    import didppy as dp

    model = dp.Model()
    v = model.add_int_var(target=0)
    model.add_transition(dp.Transition(name="increase", effects=[(v, v+1)]))
    model.add_base_case([v >= 10])
    model.dump_to_files("domain.yaml", "problem.yaml")
    reloaded_model = dp.Model.load_from_files("domain.yaml", "problem.yaml")

Notice the method :meth:`~didppy.Model.dump_to_files` works exactly same as getting the domain string and problem string from :meth:`~didppy.Model.dump_to_str`, 
then write the domain string into the file "domain.yaml" and write the problem string into the file "problem.yaml".

The method :meth:`~didppy.Model.load_from_files` works exactly same as reading the file "domain.yaml" as a single domain string and reading the file "problem.yaml" as a single problem string,
then load the model using the method :meth:`~didppy.Model.load_from_str` with the domain string and problem string.

Generated YAML files can be used with `didp-yaml <https://crates.io/crates/didp-yaml>`_, a command line tool to run DIDP solvers.
A user also can write YAML files following the `syntax <https://github.com/domain-independent-dp/didp-rs/blob/main/didp-yaml/docs/dypdl-guide.md>`_ and load them into a :class:`~didppy.Model` object.