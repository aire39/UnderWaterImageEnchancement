#!/bin/bash

if [ ! -d "libs/sfml/build_release" ] || [[ "$1" == "clear" ]]; then
rm -rf libs/sfml/build_release
mkdir -p libs/sfml/build_release
fi

pushd libs/sfml/build_release
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=../sfml-release -DBUILD_SHARED_LIBS=FALSE -DSFML_BUILD_AUDIO=FALSE -DSFML_BUILD_NETWORK=FALSE -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- -j4
make install
popd

if [ ! -d "libs/sfml/build_debug" ] || [[ "$1" == "clear" ]]; then
rm -rf libs/sfml/build_debug
mkdir -p libs/sfml/build_debug
fi

pushd libs/sfml/build_debug
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=../sfml-debug -DBUILD_SHARED_LIBS=FALSE -DSFML_BUILD_AUDIO=FALSE -DSFML_BUILD_NETWORK=FALSE -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug -- -j4
make install
popd
