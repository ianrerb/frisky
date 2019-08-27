#!/usr/bin/env python

from setuptools import find_packages, setup

DISTNAME = "frisky"
VERSION = "0.1"  # TODO add versioneer
LICENSE = "GNU GPL"
AUTHOR = "frisky Devs"
AUTHOR_EMAIL = "frisky-dev@googlegroups.com"

setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(),
)
