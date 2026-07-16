cmake_minimum_required(VERSION 3.16)
project(quoridor_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Platform-specific flags
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /EHsc")
endif()

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# Try to find pybind11 installed via pip first, then system, then fetch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_FOUND
)

if(pybind11_FOUND EQUAL 0)
    message(STATUS "Found pybind11 via pip: ${pybind11_DIR}")
    find_package(pybind11 CONFIG PATHS ${pybind11_DIR})
else()
    # Try system install
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        # Fetch from GitHub as last resort
        message(STATUS "Fetching pybind11 from GitHub...")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Create the Python module
pybind11_add_module(quoridor_cpp bindings.cpp)

# Link numpy
target_include_directories(quoridor_cpp PRIVATE ${Python3_NumPy_INCLUDE_DIRS})

# Install the module next to the source for easy import
install(TARGETS quoridor_cpp DESTINATION ${CMAKE_SOURCE_DIR})
