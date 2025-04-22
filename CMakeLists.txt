cmake_minimum_required(VERSION 3.16)

project(RebuildCCL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../ucxx/cpp/include
)

find_package(ucx REQUIRED)
find_library(UCXX_LIBRARY ucxx PATHS ${CMAKE_SOURCE_DIR}/../../ucxx/cpp/build)

add_executable(basic demo/basic.cpp)

target_link_libraries(basic PRIVATE ${UCXX_LIBRARY} ucx::ucp ucx::uct ucx::ucs)

set_target_properties(basic PROPERTIES BUILD_RPATH "$ORIGIN")