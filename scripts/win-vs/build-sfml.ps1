
if ( -not (Test-Path -Path "libs\sfml\build_vs_release") -or $args[0] -eq "clear" ) {
    rm .\libs\sfml\build_vs_release
    mkdir .\libs\sfml\build_vs_release
}

pushd .\libs\sfml\build_vs_release
cmake -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX="..\sfml-vs-release" -DBUILD_SHARED_LIBS=FALSE -DSFML_BUILD_AUDIO=FALSE -DSFML_BUILD_NETWORK=FALSE -DCMAKE_BUILD_TYPE=Release ..\
cmake --build . --target install --config Release
cmake --install . --config Release
popd

if ( -not (Test-Path -Path "libs\sfml\build_vs_debug") -or $args[0] -eq "clear" ) {
    rm .\libs\sfml\build_vs_debug
    mkdir .\libs\sfml\build_vs_debug
}

pushd .\libs\sfml\build_vs_debug
cmake -G "Visual Studio 17 2022" -DCMAKE_INSTALL_PREFIX="..\sfml-vs-debug" -DBUILD_SHARED_LIBS=FALSE -DSFML_BUILD_AUDIO=FALSE -DSFML_BUILD_NETWORK=FALSE -DCMAKE_BUILD_TYPE=Release ..\
cmake --build . --target install --config Debug
cmake --install . --config Debug
popd