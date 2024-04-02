[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![codecov](https://codecov.io/gh/gagnonanthony/CCPM/graph/badge.svg?token=7P0QUI6B8U)](https://codecov.io/gh/gagnonanthony/CCPM)
[![CI](https://github.com/gagnonanthony/CCPM/actions/workflows/build-dev.yml/badge.svg?branch=main)](https://github.com/gagnonanthony/CCPM/actions/workflows/build-dev.yml)
[![Docs](https://github.com/gagnonanthony/CCPM/actions/workflows/deploy.yml/badge.svg?branch=main)](https://github.com/gagnonanthony/CCPM/actions/workflows/build-dev.yml)

![CCPM](https://user-images.githubusercontent.com/79757265/225111405-0a5e9a60-4702-4aa7-89fc-d353124dfb63.png)

CCPM is a small toolbox to evaluate cognitive profile based on cognitive and behavioral data. 

## Installation

This library uses *poetry* to manage dependencies. To install it, use pipx with
the following command:

```
pip install pipx
pipx ensurepath
pipx install poetry
```

> [!WARNING]
> Poetry is creating is own virtual environment by default. Therefore, be sure
> to deactivate all of your virtual environment before continuing on with the
> installation.

To install CCPM and all of its dependencies, run this set of commands:

```
git clone https://github.com/gagnonanthony/CCPM.git
cd CCPM/
poetry install
```

> [!NOTE]
> The `poetry install` command will install all required dependencies as well
> as setting up a virtual environment. To access the library environment, use:
> `poetry shell`. This will activate the project's python environment in your
> current shell.
> To access your environment from other directories, use this command (from
> within the project directory), you might need to modify ~/.bashrc to your 
> specific login shell (ex: MacOS sometimes uses zsh, so ~/.zshrc):
```
ENVPATH=$(poetry env info --path)
echo "export CCPMPATH=${ENVPATH}" >> ~/.bashrc
```
> Restart your terminal. You should now be able to activate the poetry
> environment by using: `source $ENVPATH/bin/activate` from anywhere.

> [!IMPORTANT]
> ## Installing Graphviz
> Graphviz is an external dependencies required to visualize semplot from the
> ``semopy`` package used within CCPM. If you do not have Graphviz installed on
> your machine, please run the following if you are on Linux `sudo apt get graphviz`
> or `brew install graphviz` if you are on MacOS. 
