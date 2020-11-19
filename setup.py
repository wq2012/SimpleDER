"""Setup script for the package."""

import setuptools

VERSION = "0.0.3"

with open("README.md", "r") as file_object:
    LONG_DESCRIPTION = file_object.read()

SHORT_DESCRIPTION = """
A lightweight library to compute Diarization Error Rate (DER).
""".strip()

setuptools.setup(
    name="simpleder",
    version=VERSION,
    author="Quan Wang",
    author_email="quanw@google.com",
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/wq2012/SimpleDER",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
