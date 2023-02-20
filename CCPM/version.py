# -*- coding: utf-8 -*-

import glob

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 0
_version_micro = 1
_version_extra = ''

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "CCPM : Children Cognitive Profiles Mapping."
# Long description will go up on the pypi page
long_description = """
CCPM
========
CCPM is a small library containing tools to map children's cognitive
profiles based on neurocognitive tests/questionnaires.
License
=======
``CCPM`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2022, Anthony Gagnon
Université de Sherbrooke
"""

NAME = "CCPM"
MAINTAINER = "Anthony Gagnon"
MAINTAINER_EMAIL = "Anthony.Gagnon7@usherbrooke.ca"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/gagnonanthony/CCPM"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Anthony Gagnon"
AUTHOR_EMAIL = ""
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
SCRIPTS = glob.glob("Scripts/*.py")