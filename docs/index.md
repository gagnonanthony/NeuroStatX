# Welcome to NeuroStatX!

NeuroStatX is a command-line toolbox to perform statistical analysis on neuroscience data.
It had been developped mostly as part of my PhD project, which aims to understand the
relationship between the brain, cognition and behavior, hence the focus on neuroscience data.
As my project goes forward, new functionalities and scripts will be added.
**Contributions are welcome!**.

## Dependencies

Some of the statistical analyses available within NeuroStatX comes from the `semopy`
package which requires Graphviz to properly visualize semplot. If you do not
have Graphviz installed on your machine, please run the following: 
`sudo apt get graphviz` for Linux users or `brew install graphviz` for MacOS
users. **Please note that the library will work even without Graphviz if you
do not plan on using factorial analysis.**

## Installation

To install the project, please follow these steps:

```
git clone https://github.com/gagnonanthony/NeuroStatX.git
cd NeuroStatX/
poetry install
```

If you plan on developping code, please use these steps to install all the
required packages:

```
git clone https://github.com/gagnonanthony/NeuroStatX.git
cd NeuroStatX/
poetry install --with=dev,docs
```

Both of these commands will create a virtual environment. To access the
library environment, you can use: `poetry shell` from within the project
root directory. To avoid doing so each time you open a terminal, you can set
the path to the source file of the environment in your .bashrc (or equivalent).
Simply use these commands **from within the project directory**:

```
ENVPATH=$(poetry env info --path)
echo "export NeuroStatXPATH=${ENVPATH}" >> ~/.bashrc
```

Restart your terminal. You should be able to activate the project virtual
environment by using `source $NeuroStatXPATH/bin/activate`. 

## How to use?

All CLI tools should be available directly from the command line if the virtual
env is activated. Simply run `AddNodesAttributes --version` to validate.