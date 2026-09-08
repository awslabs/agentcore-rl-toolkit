#!/usr/bin/env zsh

function clone() {
    git init $1
    pushd $1
    git remote add origin $2
    git fetch --depth 1 origin $3
    git checkout FETCH_HEAD
    popd
}

# both of these packages have the same name, so we can't install
# them both at the same time. at runtime, use sys.path to pick one or the other
clone swebench https://github.com/SWE-bench/SWE-bench.git 737efd9ba02b7016feaf25660b5b14d46c0eb592
clone swegym https://github.com/SWE-Gym/SWE-Bench-Fork.git 242429c188fcfd06aad13fce9a54d450470bf0ac

# install dependencies for both
uv pip install -e ./swebench
uv pip uninstall swebench
