cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(my_add_custom LANGUAGES ASC CXX)

add_executable(demo
    add_custom.cpp
    main.cpp
)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
