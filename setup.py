"""Backward-compatibility shim for the standalone ``risk_distributions`` package.

The real code has moved to ``vivarium-risk-distributions`` (importable as
``vivarium.risk_distributions``). This empty wheel exists so that

    pip install risk_distributions

continues to resolve and pulls in the new package plus the ``vivarium-compat``
import hook. The hook redirects ``import risk_distributions`` to
``vivarium.risk_distributions`` with a ``DeprecationWarning``.

See https://github.com/ihmeuw/vivarium-suite for the new location.
"""

from setuptools import setup

setup(
    name="risk_distributions",
    description=(
        "Backward-compatibility shim. The real package is now "
        "vivarium-risk-distributions."
    ),
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    url="https://github.com/ihmeuw/vivarium-suite",
    author="The vivarium developers",
    author_email="vivarium.dev@gmail.com",
    license="BSD-3-Clause",
    classifiers=[
        "Development Status :: 7 - Inactive",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    packages=[],
    py_modules=[],
    install_requires=[
        "vivarium-risk-distributions>=3.0.0",
        "vivarium-compat>=0.4.0",
    ],
    python_requires=">=3.10",
    # Version is derived from the git tag at build time (e.g. v2.3.5 -> 2.3.5).
    # Tag, then `python -m build`, then `twine upload`.
    use_scm_version={
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
    },
    setup_requires=["setuptools_scm"],
)
