# -*- coding: utf-8 -*-

"""Top-level package for ipfx."""
import os

__author__ = """Allen Institute for Brain Science"""

version_file_path = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file_path, "r") as version_file:
    __version__ = version_file.read()
