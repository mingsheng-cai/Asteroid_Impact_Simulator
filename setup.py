"""Setup script for deepimpact package."""

import os
from setuptools import setup


def read(fname):
    """Read a file and return its contents as a string."""
    path = os.path.join(os.path.dirname(__file__), fname)
    with open(path) as f:
        return f.read()


setup(
    name="deepimpact",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    license="MIT",
    long_description=read("README.md"),
    url="https://github.com/ese-msc-2023/acs-deepimpact-atira",
    # description="""""", # TODO
    # keywords=["", "", ""], # TODO
    # author="", # TODO
    # author_email="", # TODO
    packages=["deepimpact"],
)
