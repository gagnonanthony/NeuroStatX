# CCPM

![CCPM](https://user-images.githubusercontent.com/79757265/225111405-0a5e9a60-4702-4aa7-89fc-d353124dfb63.png)

CCPM is a small toolbox to evaluate cognitive profile based on cognitive and behavioral data. 

Installation
============
To install this library, please do the following commands (virtual python
environment are recommended (e.g. virtualenv)) :

``git clone https://github.com/gagnonanthony/CCPM.git``

``cd CCPM``

``pip install -r requirements.txt``

Temporarily, install semopy separately due to the legacy use of sklearn (an upcoming PR in pip will allow
to ignore package dependencies directly in requirements.txt).

``pip install semopy==2.3.9 --no-deps``

``pip install -e .``

Installing Graphviz
===================
In order to correctly use the visualisation function of ``semopy``, it
is suggested to install Graphviz in your base system.

_FOR MAC USERS:_

``brew install graphviz``

_FOR LINUX USERS:_

``sudo apt get graphviz``

License
============
``CCPM`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2022, Anthony Gagnon,
Université de Sherbrooke
