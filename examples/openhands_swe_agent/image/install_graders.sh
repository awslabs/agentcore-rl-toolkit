#!/usr/bin/env zsh

function clone() {
    git init $1
    pushd $1
    git remote add origin $2
    git fetch --depth 1 origin $3
    git checkout FETCH_HEAD
    popd
}

# Same package name in both, so only one can be installed; runtime picks via sys.path.
clone swebench https://github.com/SWE-bench/SWE-bench.git 737efd9ba02b7016feaf25660b5b14d46c0eb592
clone swegym https://github.com/SWE-Gym/SWE-Bench-Fork.git 242429c188fcfd06aad13fce9a54d450470bf0ac

uv pip install -e ./swebench
uv pip uninstall swebench

# SWE-smith's grader (swesmith.harness.grading) -- a package, not a checkout, because the name
# collides with nothing. Every one of its requirements sits behind an extra, so this installs
# the module alone; what the grader path then imports (swebench via the sys.path append above,
# plus docker, dotenv, ghapi, unidiff) is already here from the swebench install.
uv pip install swesmith==0.0.9
