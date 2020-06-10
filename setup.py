import os
from setuptools import setup, find_packages
from distutils.cmd import Command
import glob

with open("requirements.txt", "r") as requirements_file:
    required = requirements_file.read().splitlines()

version_file_path = os.path.join(
    os.path.dirname(__file__),
    "ipfx",
    "version.txt"
)
with open(version_file_path, "r") as version_file:
    version = version_file.read()


readme_path = os.path.join(
    os.path.dirname(__file__),
    "README.md"
)
with open(readme_path, "r") as readme_file:
    readme = readme_file.read()


class CheckVersionCommand(Command):
    description = (
        "Check that this package's version matches a user-supplied version"
    )
    user_options = [
        ('expected-version=', "e", 'Compare package version against this value')
    ]

    def initialize_options(self):
        self.package_version = version
        self.expected_version = None

    def finalize_options(self):
        assert self.expected_version is not None
        if self.expected_version[0] == "v":
            self.expected_version = self.expected_version[1:]

    def run(self):
        if self.expected_version != self.package_version:
            raise ValueError(
                f"expected version {self.expected_version}, but this package "
                f"has version {self.package_version}"
            )


setup(
    name='ipfx',
    version=version,
    description="""Intrinsic Physiology Feature Extractor (IPFX) - tool for computing neuronal features from the intracellular electrophysiological recordings""",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Allen Institute for Brain Science",
    author_email="Marmot@AllenInstitute.onmicrosoft.com",
    url="https://github.com/AllenInstitute/ipfx",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=required,
    include_package_data=True,
    setup_requires=['pytest-runner'],
    keywords=["neuroscience", "bioinformatics", "scientific"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License", # Allen Institute Software License
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    cmdclass={'check_version': CheckVersionCommand}
)
