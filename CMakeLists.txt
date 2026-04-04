cmake_minimum_required(VERSION 3.11)

project(xAIInfra)


aux_source_directory(. src)
add_executable(xai ${src})