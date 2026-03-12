cmake_minimum_required(VERSION 3.16)
project(flowstate-orderbook LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only library
add_library(orderbook INTERFACE)
target_include_directories(orderbook INTERFACE include)

# --- Tests (Catch2 fetched automatically) ---
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_order_book tests/test_order_book.cpp)
target_link_libraries(test_order_book PRIVATE orderbook Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_order_book)

# --- Benchmark ---
add_executable(bench_order_book bench/bench_order_book.cpp)
target_link_libraries(bench_order_book PRIVATE orderbook)
target_compile_options(bench_order_book PRIVATE -O3)
