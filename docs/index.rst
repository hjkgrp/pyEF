.. PyEF documentation master file

.. image:: _static/logo-white.svg

PyEF: Electric Field Analysis for Molecular Systems
====================================================

.. container:: .large

   PyEF computes electric fields, electrostatic potentials, and partial charges
   from quantum mechanical calculations.


.. container:: .buttons

   `Getting Started <getting_started.html>`_
   `API Reference <api.html>`_
   `GitHub <https://github.com/hjkgrp/pyEF>`_

Overview
--------

PyEF processes molden files from QM calculations to compute:

- **Electric Fields**: At specific bonds or molecular sites
- **Electrostatic Potentials (ESP)**: At metal centers or points of interest
- **Partial Charges**: Via multiple partitioning schemes (Hirshfeld, CHELPG, Mulliken, etc.)

The package provides both a command-line interface for batch processing and a Python API
for interactive analysis.

Main Functions
--------------

All functions are methods of the ``Electrostatics`` class. Click the links for full documentation.

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Function
     - Description
     - Documentation
   * - ``getEfield()``
     - Calculate electric fields at specific bonds
     - `Guide <getting_started.html#computing-electric-fields>`_ · `API <api.html#getefield-electric-field>`_
   * - ``getESP()``
     - Calculate electrostatic potential at atomic sites
     - `Guide <getting_started.html#computing-electrostatic-potentials>`_ · `API <api.html#getesp-electrostatic-potential>`_
   * - ``getCharges()``
     - Compute partial charges for all atoms
     - `Guide <getting_started.html#computing-partial-charges>`_ · `API <api.html#getcharges-partial-charges>`_
   * - ``getElectrostatic_stabilization()``
     - Calculate electrostatic stabilization energy
     - `Guide <getting_started.html#computing-electrostatic-stabilization>`_ · `API <api.html#getelectrostatic-stabilization-electrostatic-stabilization>`_

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone git@github.com:hjkgrp/pyEF.git
   cd pyEF
   conda env create -f environment.yml
   conda activate pyef
   pip install -e .

Python API Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics

   # Initialize
   es = Electrostatics(['optim.molden'], ['optim.xyz'])

   # Calculate electric field at a bond
   df = es.getEfield('Hirshfeld_I', 'output', '/path/to/multiwfn',
                     input_bond_indices=[(25, 26)])

   # For ESP, initialize with esp_atom_idx
   es_esp = Electrostatics(['optim.molden'], ['optim.xyz'], esp_atom_idx=[30])
   esp_df = es_esp.getESP('Hirshfeld_I', 'esp_output', '/path/to/multiwfn')

Documentation
-------------

.. toctree::
   :maxdepth: 2

   getting_started
   api

Citation
--------

.. code-block:: bibtex

   @software{pyef,
     title = {PyEF: Electric Field Analysis for Molecular Systems},
     author = {Manetsch, Melissa and Kastner, David W.},
     year = {2025},
     url = {https://github.com/hjkgrp/pyEF}
   }

License
-------

PyEF is released under the MIT License.

**Authors:** Melissa Manetsch and David W. Kastner
