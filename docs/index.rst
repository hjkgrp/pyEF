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
   `GitHub <https://github.com/davidkastner/pyef>`_

Overview
--------

PyEF processes molden files from QM calculations to compute:

- **Electric Fields**: At specific bonds or molecular sites
- **Electrostatic Potentials (ESP)**: At metal centers or points of interest
- **Partial Charges**: Via multiple partitioning schemes (Hirshfeld, CHELPG, Mulliken, etc.)

The package provides both a command-line interface for batch processing and a Python API
for interactive analysis.

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   git clone git@github.com:davidkastner/pyEF.git
   cd pyEF
   ./install.sh

Python API Example
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics

   # Initialize
   es = Electrostatics(['optim.molden'], ['optim.xyz'])

   # Calculate electric field at a bond
   df = es.getEfield('Hirshfeld_I', 'output', '/path/to/multiwfn',
                     input_bond_indices=[(25, 26)])

   # Calculate ESP at metal center
   esp_df = es.getESP('Hirshfeld_I', 'esp_output', '/path/to/multiwfn')

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
     url = {https://github.com/davidkastner/pyef}
   }

License
-------

PyEF is released under the MIT License.

**Authors:** Melissa Manetsch and David W. Kastner
