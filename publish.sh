#!/bin/bash
set -e

echo "Do we have docker?"
docker --version || echo "No. Please install it."
echo "Looks fine"

echo "Looking for manylinux build image"
if docker images | grep -q maturin-manylinux
then
    echo "maturin-manylinux is here. We're good."
else
    echo "Not there, need to build it. It'll take a minute"
    mkdir maturin-build-temp
    cd maturin-build-temp
    git clone https://github.com/PyO3/maturin.git
    cd maturin
    docker build -t maturin-manylinux -f maturin-manylinux.dockerfile .
    echo "Built the manylinux build image"
fi


# It's not super clear why Maturin expects this file is here
cp Cargo.toml Cargo.toml.orig

echo "Now building stoicheia as planned"
docker run -it --rm -v $(pwd):/io maturin-manylinux \
    maturin publish --cargo-extra-args="--features python"

rm Cargo.toml.orig
echo "Looks like we're done here."