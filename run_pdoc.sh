#!/bin/bash
set -o errexit

# Get project path.
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd ${PROJECT_PATH}

rm -r docs

# This script requires pdoc:
# pip3 install pdoc
python3 -m pdoc simpleder -o docs

popd
