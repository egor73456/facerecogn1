cmake_minimum_required(VERSION 3.10.0)

project(Homework CXX)
set(CMAKE_CXX_STANDARD 17)
include(cmake/StandardProjectSettings.cmake)
include(cmake/Catch.cmake)
# include(FetchContent)
set(OpenCV_DIR "C:\\opencv\\build")

#git submodule add https://github.com/arsenm/sanitizers-cmake.git external/sanitizers-cmake

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

# Link this 'library' to set the c++ standard / compile-time options requested
find_package(Threads)
add_library(default INTERFACE)
target_compile_features(default INTERFACE cxx_std_17)

# include_directories(external/opencv/include)
find_package(OpenCV REQUIRED)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(default)

enable_testing()
add_subdirectory(external/catch2)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/sanitizers-cmake/cmake")
add_subdirectory(external/sanitizers-cmake)
# add_subdirectory(external/opencv)
add_subdirectory(src)
