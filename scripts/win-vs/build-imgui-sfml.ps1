
$sfml_path="$(pwd)\libs/sfml\sfml-vs-"
$imgui_path="$(pwd)\libs\imgui"

if ( -not (Test-Path -Path "libs\imgui-sfml\imgui-sfml-vs-release") -or $args[0] -eq "clear" ) {
    rm .\libs\imgui-sfml\build_vs_release
    mkdir .\libs\imgui-sfml\build_vs_release
}

pushd .\libs\imgui-sfml\build_vs_release
cmake -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX="..\imgui-sfml-vs-release" -DBUILD_SHARED_LIBS=FALSE -DSFML_DIR="${sfml_path}release\lib\cmake\SFML" -DIMGUI_DIR="${imgui_path}" -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --target install --config Release
cmake --install . --config Release
popd

if ( -not (Test-Path -Path "libs\imgui-sfml\imgui-sfml-vs-debug") -or $args[0] -eq "clear" ) {
    rm .\libs\imgui-sfml\build_vs_debug
    mkdir .\libs\imgui-sfml\build_vs_debug
}

pushd .\libs\imgui-sfml\build_vs_debug
cmake -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX="..\imgui-sfml-vs-debug" -DBUILD_SHARED_LIBS=FALSE -DSFML_DIR="${sfml_path}debug\lib\cmake\SFML" -DIMGUI_DIR="${imgui_path}" -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --target install --config Debug
cmake --install . --config Debug
popd