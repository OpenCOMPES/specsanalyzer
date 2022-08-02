# specsanalyzer
This is a package to import and convert MCP analyzer images from SPECS Phoibos analyzers into energy and emission angle/physical coordinates.

## Getting started

You should create a virtual environment. This is optional, but highly recommended as
the required pypi packages might require many dependencies with specific versions
that might conflict with other libraries that you have installed. This was tested
with Python 3.8.

If you don't have Python 3.8 installed on your computer, follow these commands:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8 python3-dev libpython3.8-dev python-numpy python3-pip
sudo pip install --upgrade pip
sudo pip install virtualenv
```

You can now install your virtual environment with python3.8 interpreter

```
mkdir <your-brand-new-folder>
cd <your-brand-new-folder>
virtualenv --python=python3.8 .pyenv
source .pyenv/bin/activate
```

Install the specsanalyzer package:
```
pip install --upgrade pip
pip install specsanalyzer
```

Development installation:
```
pip install --upgrade pip
git clone git@github.com:mpes-kit/specsanalyzer.git
cd specsanalyzer
pip install -e .
```
