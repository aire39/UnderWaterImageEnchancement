cmake_minimum_required(VERSION 3.15)
project(underwater-image-enchancement)

## check if debug or release

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(use_build_type "debug")
    add_compile_options(-ggdb -O0)
else ()
	set(use_build_type "release")
    add_compile_options(-ggdb -O3)
endif ()

## compiler standard and other settings

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

## command scripts to build the libraries before compiling project

# will rebuild the libraries
set(CLEAR_LIBS "")
if (${CLEAR_SFML_LIBS})
	set(CLEAR_LIBS "clear")
endif()

if (DEFINED WIN32)
        find_program(FOUND_GIT "git")

        if (FOUND_GIT)
        execute_process(
                        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/libs/sfml"
                        COMMAND git apply ${PROJECT_SOURCE_DIR}/libs/patches/sfml/onresize_win32_2.6.0.patch
                       )
        else()
        message("Unable to find git command! Will not try to apply patches.")
        endif()
endif()

if (DEFINED WIN32 AND (DEFINED MSYS OR DEFINED MINGW))
message("Using MSYS or MinGW")

execute_process(
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/win-mingw/build-sfml.sh ${CLEAR_LIBS}
               )

execute_process(
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/win-mingw/build-imgui-sfml.sh ${CLEAR_LIBS}
               )

elseif (DEFINED UNIX)
	message("Using Linux")

execute_process(
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/linux/build-sfml.sh ${CLEAR_LIBS}
               )

execute_process(
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		COMMAND bash ${PROJECT_SOURCE_DIR}/scripts/linux/build-imgui-sfml.sh ${CLEAR_LIBS}
               )


elseif(DEFINED WIN32 AND DEFINED MSVC)
message("Using Visual Studio")

execute_process(
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		COMMAND powershell ${PROJECT_SOURCE_DIR}/scripts/win-vs/build-sfml.ps1 ${CLEAR_LIBS}
               )

execute_process(
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		COMMAND powershell ${PROJECT_SOURCE_DIR}/scripts/win-vs/build-imgui-sfml.ps1 ${CLEAR_LIBS}
               )

else()

message("No supported platform available!")
return(-1)

endif()

## location of libraries

set(SFML_STATIC_LIBRARIES TRUE)

if (DEFINED WIN32 AND (DEFINED MSYS OR DEFINED MINGW))

set(SFML_DIR "${PROJECT_SOURCE_DIR}/libs/sfml/sfml-${use_build_type}/lib/cmake/SFML")
set(ImGui-SFML_DIR "${PROJECT_SOURCE_DIR}/libs/imgui-sfml/imgui-sfml-${use_build_type}/lib/cmake/ImGui-SFML")

elseif (DEFINED UNIX)

set(SFML_DIR "${PROJECT_SOURCE_DIR}/libs/sfml/sfml-${use_build_type}/lib/cmake/SFML")
set(ImGui-SFML_DIR "${PROJECT_SOURCE_DIR}/libs/imgui-sfml/imgui-sfml-${use_build_type}/lib/cmake/ImGui-SFML")


elseif(DEFINED WIN32 AND DEFINED MSVC)

set(SFML_DIR "${PROJECT_SOURCE_DIR}/libs/sfml/sfml-vs-${use_build_type}/lib/cmake/SFML")
set(ImGui-SFML_DIR "${PROJECT_SOURCE_DIR}/libs/imgui-sfml/imgui-sfml-vs-${use_build_type}/lib/cmake/ImGui-SFML")

else()

message("No supported platform available!")
return(-1)

endif()


find_package(Threads REQUIRED)
find_package(SFML 2 COMPONENTS system window graphics REQUIRED)
find_package(ImGui-SFML REQUIRED)

add_subdirectory(libs/cli11)
add_subdirectory(libs/spdlog)

include_directories(${PROJECT_SOURCE_DIR}/libs)
file(GLOB TINYDIALOG ${PROJECT_SOURCE_DIR}/libs/tinyfiledialogs/tinyfiledialogs.c ${PROJECT_SOURCE_DIR}/libs/tinyfiledialogs/tinyfiledialogs.h)
file(GLOB IMPLOT ${PROJECT_SOURCE_DIR}/libs/implot/implot.h ${PROJECT_SOURCE_DIR}/libs/implot/implot.cpp ${PROJECT_SOURCE_DIR}/libs/implot/implot_internal.h ${PROJECT_SOURCE_DIR}/libs/implot/implot_items.cpp)
file(GLOB COMMON ${PROJECT_SOURCE_DIR}/common/*)

## files needed to build project

include_directories(${PROJECT_SOURCE_DIR})

add_executable(underwater-image-enchancement
               main.cpp
               imageops/colormodel.cpp
               imageops/colormodel.h
               ${TINYDIALOG}
               ${IMPLOT}
               ${COMMON}
              )

target_link_libraries(underwater-image-enchancement
                      sfml-window
                      sfml-graphics
                      ImGui-SFML::ImGui-SFML
                      Threads::Threads
                      CLI11::CLI11
                      spdlog
                     )

## copy files after build to the directory of the output (if needed)

if (DEFINED WIN32 AND (DEFINED MSYS OR DEFINED MINGW))

add_custom_command(TARGET underwater-image-enchancement POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $ENV{MINGW_PREFIX}/bin/libgcc_s_seh-1.dll $ENV{MINGW_PREFIX}/bin/libwinpthread-1.dll $ENV{MINGW_PREFIX}/bin/libstdc++-6.dll .
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR} )
endif()
