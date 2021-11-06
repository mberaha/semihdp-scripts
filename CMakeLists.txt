cmake_minimum_required(VERSION 3.13.0)

project(semihdp_scripts)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -ftree-vectorize")

add_subdirectory(lib/bayesmix)
add_definitions(-w)

# Build run executable
add_library(semihdp_scripts OBJECT)
target_sources(semihdp_scripts
  PUBLIC 
   utils.hpp 
   utils.cpp
) 

add_executable(run_from_file $<TARGET_OBJECTS:bayesmix> 
  run_from_file.cpp utils.hpp utils.cpp)
target_include_directories(run_from_file PUBLIC ${BAYESMIX_INCLUDE_PATHS})
target_link_libraries(run_from_file PUBLIC ${BAYESMIX_LINK_LIBRARIES})
target_compile_options(run_from_file PUBLIC ${BAYESMIX_COMPILE_OPTIONS})

add_executable(simu1 $<TARGET_OBJECTS:bayesmix> 
  run_simulation1.cpp utils.hpp utils.cpp)
target_include_directories(simu1 PUBLIC ${BAYESMIX_INCLUDE_PATHS})
target_link_libraries(simu1 PUBLIC ${BAYESMIX_LINK_LIBRARIES})
target_compile_options(simu1 PUBLIC ${BAYESMIX_COMPILE_OPTIONS})

add_executable(simu2 $<TARGET_OBJECTS:bayesmix> run_simulation2.cpp)
target_include_directories(simu2 PUBLIC ${INCLUDE_PATHS})
target_link_libraries(simu2 PUBLIC ${LINK_LIBRARIES})
target_compile_options(simu2 PUBLIC ${COMPILE_OPTIONS})

add_executable(simu100 $<TARGET_OBJECTS:bayesmix> run_simulation_many.cpp)
target_include_directories(simu100 PUBLIC ${INCLUDE_PATHS})
target_link_libraries(simu100 PUBLIC ${LINK_LIBRARIES})
target_compile_options(simu100 PUBLIC ${COMPILE_OPTIONS})
