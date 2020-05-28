import os
from setuptools import setup, find_packages

with open("requirements.txt", "r") as requirements_file:
    required = requirements_file.read().splitlines()

version_file_path = os.path.join(
    os.path.dirname(__file__),
    "ipfx",
    "version.txt"
)
with open(version_file_path, "r") as version_file:
    version = version_file.read()

setup(
    name='ipfx',
    version=version,
    description="""intrinsic physiology feature extractor""",
    author="David Feng",
    author_email="Marmot@AllenInstitute.onmicrosoft.com",
    url="https://github.com/AllenInstitute/ipfx",
    packages=find_packages(),
    install_requires=required,
    include_package_data=True,
    setup_requires=['pytest-runner'],
)
