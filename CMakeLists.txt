cmake_minimum_required(VERSION 3.11)

project(xAIInfra)


# Find OpenMP package
find_package(OpenMP REQUIRED)

aux_source_directory(. src)
add_executable(xai ${src})
set_target_properties(xai PROPERTIES
    CXX_STANDARD 23 # Options: 98, 11, 14, 17, 20, 23, 26
    CXX_STANDARD_REQUIRED ON # Force an error if the compiler doesn't support it
)

target_link_libraries(xai PUBLIC OpenMP::OpenMP_CXX)