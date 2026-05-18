==================
Risk Distributions
==================

**NOTE: This repository has been archived.**

The ``risk_distributions`` package has been renamed and migrated into the
`vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

What changed
------------

- **PyPI distribution:** ``risk_distributions`` -> ``vivarium-risk-distributions``
- **Import path:** ``risk_distributions`` -> ``vivarium.risk_distributions``
- **Source:** ``ihmeuw/risk_distributions`` (archived) ->
  ``ihmeuw/vivarium-suite`` (under ``libs/risk-distributions/``)
- **Docs:** https://vivarium-risk-distributions.readthedocs.io/

What this last release (v2.4.0) is
----------------------------------

A backward-compatibility shim. It contains no code; installing it pulls in:

- ``vivarium-risk-distributions>=3.0.0`` - the real implementation under the
  new import path.
- ``vivarium-compat>=0.4.0`` - an import hook that lets ``import risk_distributions``
  continue to work, emitting a ``DeprecationWarning``. Update your imports before
  the hook is removed.

If you depend on a specific pre-rename version (e.g. ``risk_distributions==2.3.4``),
pin to that version - those earlier releases still ship the original module
and continue to install standalone.

To migrate fully to the new package
-----------------------------------

**Install:**

.. code-block:: bash

   pip install vivarium-risk-distributions   # was: pip install risk_distributions

**Import:**

.. code-block:: python

   import vivarium.risk_distributions                       # was: import risk_distributions

Original package overview
=========================

.. image:: https://badge.fury.io/py/risk_distributions.svg
    :target: https://badge.fury.io/py/risk-distributions

.. image:: https://github.com/ihmeuw/risk_distributions/actions/workflows/build.yml/badge.svg?branch=main
    :target: https://github.com/ihmeuw/risk_distributions
    :alt: Latest Version

.. image:: https://readthedocs.org/projects/risk-distributions/badge/?version=latest
    :target: https://risk-distributions.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

This library contains various probability distributions, compatible for use with the ``Vivarium`` framework

You can install ``risk_distributions`` from PyPI with pip:

  ``> pip install risk_distributions``

or build it from source with

  ``> git clone https://github.com/ihmeuw/risk_distributions.git``

  ``> cd risk_distributions``

  ``> python setup.py install``


`Check out the docs! <https://risk-distributions.readthedocs.io/en/latest/>`_
-----------------------------------------------------------------------------
