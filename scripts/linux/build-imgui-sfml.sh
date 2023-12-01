#!/bin/bash

sfml_path="$(pwd)/libs/sfml/sfml-"
imgui_path="$(pwd)/libs/imgui"

if [ ! -d "libs/imgui-sfml/imgui-sfml-release" ] || [[ "$1" == "clear" ]]; then
rm -rf libs/imgui-sfml/build_release
mkdir -p libs/imgui-sfml/build_release
fi

pushd libs/imgui-sfml/build_release
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=../imgui-sfml-release -DBUILD_SHARED_LIBS=FALSE -DSFML_DIR="${sfml_path}release/lib/cmake/SFML" -DIMGUI_DIR="${imgui_path}" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release -- -j4
make install
popd

if [ ! -d "libs/imgui-sfml/imgui-sfml-debug" ] || [[ "$1" == "clear" ]]; then
rm -rf libs/imgui-sfml/build_debug
mkdir -p libs/imgui-sfml/build_debug
fi

pushd libs/imgui-sfml/build_debug
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=../imgui-sfml-debug -DBUILD_SHARED_LIBS=FALSE -DSFML_DIR="${sfml_path}debug/lib/cmake/SFML" -DIMGUI_DIR="${imgui_path}" -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug -- -j4
make install
popd
