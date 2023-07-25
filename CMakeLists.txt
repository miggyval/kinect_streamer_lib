cmake_minimum_required( VERSION 3.10 )

project( kinect_streamer_lib )

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(kinect_streamer_lib_VERSION 1.0)

include(GenerateExportHeader)

set( freenect2_DIR $ENV{HOME}/Libraries/freenect2/lib/cmake/freenect2)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

find_package( freenect2 REQUIRED )
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )
find_package( OpenPose REQUIRED )
find_package( gflags REQUIRED )
find_package( PCL REQUIRED )
find_package( pcl_conversions )

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${freenect2_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${GTK3_INCLUDE_DIRS}
    $ENV{HOME}/argparse/include
    ${PCL_INCLUDE_DIRS}
)

include_directories(
    ${OpenPose_INCLUDE_DIRS}
    ${GFLAGS_INCLUDE_DIR}
    ${freenect2_INCLUDE_DIR}
)


cuda_add_library( kinect_streamer src/kinect_streamer_cuda.cu)
add_executable( kinect_demo	src/kinect_demo.cpp )
add_executable( kinect_openpose_demo src/kinect_openpose_demo.cpp )

add_library( kinect_streamer_lib SHARED src/kinect_streamer.cpp )

link_directories(${PCL_LIBRARY_DIRS})

target_link_libraries( kinect_demo ${freenect2_LIBRARIES} ${OpenCV_LIBS} kinect_streamer_lib stdc++fs gflags )
target_link_libraries( kinect_openpose_demo ${freenect2_LIBRARIES} ${OpenCV_LIBS} ${OpenPose_LIBS} kinect_streamer_lib stdc++fs gflags )
target_link_libraries( kinect_streamer_lib ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${freenect2_LIBRARIES} ${OpenCV_LIBS} ${GTK3_LIBRARIES} ${CUDA_LIBRARIES} stdc++fs )

set_target_properties( kinect_demo	PROPERTIES CXX_STANDARD 17 )
set_target_properties( kinect_openpose_demo	PROPERTIES CXX_STANDARD 17 )

add_compile_options(${GTK3_CFLAGS_OTHER})

install(
  TARGETS kinect_streamer_lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/kinect_streamer
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/kinect_streamer
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/kinect_streamer
)
