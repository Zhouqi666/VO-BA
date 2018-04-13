cmake_minimum_required( VERSION 2.8 )


project (VO_BA)

SET(CMAKE_BUILD_TYPE Release)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#use MRPT
find_package( MRPT REQUIRED graphs;gui;opengl;maps)

#use sophus
find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS})

#use Eigen
include_directories( ${EIGEN3_INCLUDE_DIRS})

#use g2o
LIST( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
SET( G2O_ROOT /usr/local/include/g2o )
FIND_PACKAGE( G2O )
include_directories( ${G2O_INCLUDE_DIR})
SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

#use include
INCLUDE_DIRECTORIES(
${PROJECT_SOURCE_DIR}/include
)

# CSparse
FIND_PACKAGE( CSparse )
INCLUDE_DIRECTORIES( ${CSPARSE_INCLUDE_DIR} )

# CHolomd
Find_Package(Cholmod REQUIRED)
INCLUDE_DIRECTORIES( ${CHOLMOD_INCLUDE_DIR} )

#ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#use opencv
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

#use pangolin
find_package(Pangolin REQUIRED)
include_directories( ${Pangolin_INCLUDE_DIRS} )

#src/camera.cpp src/frame.cpp src/map.cpp src/mappoint.cpp src/vo.cpp
set(SRC_LIST src/camera.cpp src/frame.cpp src/map.cpp src/mappoint.cpp src/vo.cpp)

add_executable(VO_BA ${SRC_LIST})
target_link_libraries( ${PROJECT_NAME}
${MRPT_LIBS}
${Sophus_LIBRARIES}
${EIGEN3_LIBS}
${G2O_LIBS}
${CHOLMOD_LIBRARIES}
${CERES_LIBRARIES}
${OpenCV_LIBRARIES}
${Pangolin_LIBRARIES})


