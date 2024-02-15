# Welcome to CCPM!

Children Cognitive Profiles Mapping (CCPM) is a small toolbox allowing for the
creation, evaluating and modelisation of cognitive and behavioral profiles.
It leverages fuzzy clustering, graph theory and statistical approaches to
investigate links between cognition/behavior and neurophysiology. 

## Dependencies

Some of the statistical analyses available within CCPM comes from the `semopy`
package which requires Graphviz to properly visualize semplot. If you do not
have Graphviz installed on your machine, please run the following: 
`sudo apt get graphviz` for Linux users or `brew install graphviz` for MacOS
users. **Please note that the library will work even without Graphviz if you
do not plan on using factorial analysis.**

## Installation

To install the project, please follow these steps:

```
git clone https://github.com/gagnonanthony/CCPM.git
cd CCPM/
poetry install
```

If you plan on developping code, please use these steps to install all the
required packages:

```
git clone https://github.com/gagnonanthony/CCPM.git
cd CCPM/
poetry install --with=dev,docs
```

Both of these commands will create a virtual environment. To access the
library environment, you can use: `poetry shell` from within the project
root directory. To avoid doing so each time you open a terminal, you can set
the path to the source file of the environment in your .bashrc (or equivalent).
Simply use these commands **from within the project directory**:

```
ENVPATH=$(poetry env info --path)
echo "export CCPMPATH=${ENVPATH}" >> ~/.bashrc
```

Restart your terminal. You should be able to activate the project virtual
environment by using `source $ENVPATH/bin/activate`. 
