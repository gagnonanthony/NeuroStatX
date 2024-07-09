[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
![Python](https://img.shields.io/badge/Python-3.10/3.11-blue)
[![codecov](https://codecov.io/gh/gagnonanthony/NeuroStatX/graph/badge.svg?token=7P0QUI6B8U)](https://codecov.io/gh/gagnonanthony/NeuroStatX)
[![CI](https://github.com/gagnonanthony/NeuroStatX/actions/workflows/build-dev.yml/badge.svg?branch=main)](https://github.com/gagnonanthony/NeuroStatX/actions/workflows/build-dev.yml)
[![Docs](https://github.com/gagnonanthony/NeuroStatX/actions/workflows/deploy.yml/badge.svg?branch=main)](https://github.com/gagnonanthony/NeuroStatX/actions/workflows/build-dev.yml)

![logo_with_text](https://github.com/gagnonanthony/NeuroStatX/assets/79757265/def2209e-e494-4427-9c3c-b8349c0c289b)


NeuroStatX is a command-line toolbox to perform statistical analysis on
neuroscience data. It had been developped mostly as part of my PhD project,
which aims to understand the relationship between the brain, cognition and
behavior, hence the focus on neuroscience data. As my project goes forward,
new functionalities and scripts will be added. **Contributions are welcome!**.

> [!NOTE] 
> NeuroStatX also offers a strong testing infrastructure to ensure robust and
> reproducible results when applicable. Unit test are already implemented for
> most functions, and CLI script are tested to ensure proper execution.

## Installation

### Through PiPy.

> [!WARNING]
> NeuroStatX will become available through PiPy once release 1.0.0 is out!
> Stay tuned.

### From source.

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

To install NeuroStatX and all of its dependencies, run this set of commands:

```
git clone https://github.com/gagnonanthony/NeuroStatX.git
cd NeuroStatX/
poetry install
```

> [!NOTE]
> The `poetry install` command will install all required dependencies as well
> as setting up a virtual environment. To access the library environment, use:
> `poetry shell` from the project root directory. This will activate the
> project's python environment in your current shell.
> To access your environment from other directories, use this command (from
> within the project directory), you might need to modify ~/.bashrc to your 
> specific login shell (ex: MacOS sometimes uses zsh, so ~/.zshrc or
> ~/.zprofile):
```
ENVPATH=$(poetry env info --path)
echo "export NeuroStatXPATH=${ENVPATH}" >> ~/.bashrc
```
Restart your terminal. You should now be able to activate the poetry
environment by using: `source $NeuroStatXPATH/bin/activate` from anywhere.

> [!IMPORTANT]
> ## Installing Graphviz
> Graphviz is an external dependencies required to visualize semplot from the
> ``semopy`` package used within NeuroStatX. If you do not have Graphviz
> installed on your machine, please run the following if you are on Linux
> `sudo apt get graphviz` or `brew install graphviz` if you are on MacOS.
