Risk Distributions
======================

**NOTE: This repository is archived and will receive no further updates.**

The ``risk_distributions`` package has been renamed and migrated into the
`vivarium-suite monorepo <https://github.com/ihmeuw/vivarium-suite>`_.

What changed
------------

- **PyPI distribution:** ``risk_distributions`` -> ``vivarium-risk-distributions``
- **Import path:** ``risk_distributions`` -> ``vivarium.risk_distributions``
- **Source:** ``ihmeuw/risk_distributions`` (archived) ->
  ``ihmeuw/vivarium-suite`` (under ``libs/risk-distributions/``)
- **Docs:** https://vivarium-risk-distributions.readthedocs.io/

The final release on PyPI (``v2.3.5``) ships the same code as ``v2.3.4`` with
this banner added. ``v2.4.0`` shipped a metapackage/shim that has since been
yanked; the archived repo's final release is ``v2.3.5``. ``pip install risk_distributions``
will keep resolving and importing as it did in ``v2.3.4``, but the code is frozen
and will not receive updates.

To migrate fully to the new package
-----------------------------------

**Install:**

.. code-block:: bash

    pip install vivarium-risk-distributions  # was: pip install risk_distributions

**Import:**

.. code-block:: python

    import vivarium.risk_distributions  # was: import risk_distributions

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
