#!/bin/sh
set -e
maturin develop --cargo-extra-args="--features python"
py.test stoicheia