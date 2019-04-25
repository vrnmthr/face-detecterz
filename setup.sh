#!/usr/bin/env bash

dir = $PWD
mkdir ~/Source/

# install lua Torch to ~/Source/torch
cd ~/Source/
mkdir torch && cd torch
git clone https://github.com/torch/distro.git --recursive
cd distro
bash install-deps
./install.sh
# add to path and add dependencies
echo 'export PATH=$PATH:~/Source/torch/distro/install/bin' >> ~/.bash_profile
source ~/.bash_profile
for NAME in dpnn nn optim csvigo cutorch cunn; do luarocks install $NAME; done

# install openface to ~/Source/openface
cd ~/Source/
brew install findutils
brew install coreutils
git clone https://github.com/cmusatyalab/openface.git --recursive
cd openface
python2 setup.py install
./models/get-models.sh
