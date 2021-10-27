cmake_minimum_required(VERSION 3.13.0)
project(test_bayesmix)

message(${INCLUDE_PATHS})
# Build run executable
target_sources(bayesmix
  PUBLIC 
   utils.hpp 
   utils.cpp
) 

add_executable(simu1 $<TARGET_OBJECTS:bayesmix-objs> run_simulation1.cpp)
target_include_directories(simu1 PUBLIC ${INCLUDE_PATHS})
target_link_libraries(simu1 PUBLIC ${LINK_LIBRARIES})
target_compile_options(simu1 PUBLIC ${COMPILE_OPTIONS})


add_executable(simu2 $<TARGET_OBJECTS:bayesmix-objs> run_simulation2.cpp)
target_include_directories(simu2 PUBLIC ${INCLUDE_PATHS})
target_link_libraries(simu2 PUBLIC ${LINK_LIBRARIES})
target_compile_options(simu2 PUBLIC ${COMPILE_OPTIONS})

add_executable(simu100 $<TARGET_OBJECTS:bayesmix-objs> run_simulation_many.cpp)
target_include_directories(simu100 PUBLIC ${INCLUDE_PATHS})
target_link_libraries(simu100 PUBLIC ${LINK_LIBRARIES})
target_compile_options(simu100 PUBLIC ${COMPILE_OPTIONS})


add_executable(chile $<TARGET_OBJECTS:bayesmix-objs> run_chile.cpp)
target_include_directories(chile PUBLIC ${INCLUDE_PATHS})
target_link_libraries(chile PUBLIC ${LINK_LIBRARIES})
target_compile_options(chile PUBLIC ${COMPILE_OPTIONS})
