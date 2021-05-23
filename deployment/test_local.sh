#!/bin/bash
# install_locally.sh
# Install locally for testing

cd ../

# Remove the env dir if it exists
if [ -d "env" ]; then rm -rf env; fi

virtualenv -p python3 env
. env/bin/activate
pip install dist/objectdaddy-*.tar.gz
pip install rtsparty ipython

ipython
