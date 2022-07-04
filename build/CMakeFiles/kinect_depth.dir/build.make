# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/medrobotics/kinect_streamer_lib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/medrobotics/kinect_streamer_lib/build

# Include any dependencies generated for this target.
include CMakeFiles/kinect_depth.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kinect_depth.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kinect_depth.dir/flags.make

CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o: CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o.depend
CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o: CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o.cmake
CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o: ../src/kinect_depth.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/medrobotics/kinect_streamer_lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o"
	cd /home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/src && /usr/bin/cmake -E make_directory /home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/src/.
	cd /home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/src && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/src/./kinect_depth_generated_kinect_depth.cu.o -D generated_cubin_file:STRING=/home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/src/./kinect_depth_generated_kinect_depth.cu.o.cubin.txt -P /home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o.cmake

# Object files for target kinect_depth
kinect_depth_OBJECTS =

# External object files for target kinect_depth
kinect_depth_EXTERNAL_OBJECTS = \
"/home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o"

libkinect_depth.so: CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o
libkinect_depth.so: CMakeFiles/kinect_depth.dir/build.make
libkinect_depth.so: /usr/local/cuda-11.1/lib64/libcudart_static.a
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/librt.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_common.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_io.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libfreetype.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libz.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpng.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libtiff.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libexpat.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
libkinect_depth.so: /opt/ros/noetic/lib/libroscpp.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpthread.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
libkinect_depth.so: /opt/ros/noetic/lib/librosconsole.so
libkinect_depth.so: /opt/ros/noetic/lib/librosconsole_log4cxx.so
libkinect_depth.so: /opt/ros/noetic/lib/librosconsole_backend_interface.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
libkinect_depth.so: /opt/ros/noetic/lib/libxmlrpcpp.so
libkinect_depth.so: /opt/ros/noetic/lib/libroscpp_serialization.so
libkinect_depth.so: /opt/ros/noetic/lib/librostime.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
libkinect_depth.so: /opt/ros/noetic/lib/libcpp_common.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_people.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libqhull.so
libkinect_depth.so: /usr/lib/libOpenNI.so
libkinect_depth.so: /usr/lib/libOpenNI2.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libfreetype.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libz.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libjpeg.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpng.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libtiff.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libexpat.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
libkinect_depth.so: /usr/local/cuda-11.1/lib64/libcudart_static.a
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/librt.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_common.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_io.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL2-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-7.1.so.7.1p.1
libkinect_depth.so: /opt/ros/noetic/lib/libroscpp.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpthread.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.71.0
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.71.0
libkinect_depth.so: /opt/ros/noetic/lib/librosconsole.so
libkinect_depth.so: /opt/ros/noetic/lib/librosconsole_log4cxx.so
libkinect_depth.so: /opt/ros/noetic/lib/librosconsole_backend_interface.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/liblog4cxx.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.71.0
libkinect_depth.so: /opt/ros/noetic/lib/libxmlrpcpp.so
libkinect_depth.so: /opt/ros/noetic/lib/libroscpp_serialization.so
libkinect_depth.so: /opt/ros/noetic/lib/librostime.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.71.0
libkinect_depth.so: /opt/ros/noetic/lib/libcpp_common.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.71.0
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.71.0
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so.0.4
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libqhull.so
libkinect_depth.so: /usr/lib/libOpenNI.so
libkinect_depth.so: /usr/lib/libOpenNI2.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_features.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_search.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_io.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libpcl_common.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkalglib-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOXML-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libfreetype.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkIOImage-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtksys-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libvtkmetaio-7.1.so.7.1p.1
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libz.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libGLEW.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libSM.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libICE.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libX11.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libXext.so
libkinect_depth.so: /usr/lib/x86_64-linux-gnu/libXt.so
libkinect_depth.so: CMakeFiles/kinect_depth.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/medrobotics/kinect_streamer_lib/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libkinect_depth.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kinect_depth.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kinect_depth.dir/build: libkinect_depth.so

.PHONY : CMakeFiles/kinect_depth.dir/build

CMakeFiles/kinect_depth.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kinect_depth.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kinect_depth.dir/clean

CMakeFiles/kinect_depth.dir/depend: CMakeFiles/kinect_depth.dir/src/kinect_depth_generated_kinect_depth.cu.o
	cd /home/medrobotics/kinect_streamer_lib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/medrobotics/kinect_streamer_lib /home/medrobotics/kinect_streamer_lib /home/medrobotics/kinect_streamer_lib/build /home/medrobotics/kinect_streamer_lib/build /home/medrobotics/kinect_streamer_lib/build/CMakeFiles/kinect_depth.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kinect_depth.dir/depend

