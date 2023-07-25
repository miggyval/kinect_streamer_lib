#include <iostream>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <cstdio>
#include <cmath>
#include <cstring>

#include <kinect_streamer/kinect_streamer.hpp>

/**
 * @brief The KinectStreamer namespace
 * 
 */
namespace KinectStreamer {

/**
 * @brief Construct a Kinect Device with a given serial number
 * 
 * @param serial Serial Number of the Kinect v2
 */
KinectDevice::KinectDevice(std::string serial) {

    pipeline = new libfreenect2::OpenGLPacketPipeline();

    freenect2 = new libfreenect2::Freenect2();
    listener = new libfreenect2::SyncMultiFrameListener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
    
    if (freenect2->enumerateDevices() == 0) {
        std::cout << "No devices found!" << std::endl;
        throw std::exception();
    }
    if (serial == "") {
        serial = freenect2->getDefaultDeviceSerialNumber();
    }
    dev = freenect2->openDevice(serial, pipeline);
    dev->setColorFrameListener(listener);
    dev->setIrAndDepthFrameListener(listener);
}


/**
 * @brief Set the intrinsic parameters of the RGB camera
 * 
 * @param cx    The x-axis optical center of the camera in pixels.
 * @param cy    The y-axis optical center of the camera in pixels.
 * @param fx    The x-axis focal length of the camera in pixels.
 * @param fy    The y-axis focal length of the camera in pixels.
 */
void KinectDevice::set_color_params(float cx, float cy, float fx, float fy) {
    color_params.cx = cx;
    color_params.cy = cy;
    color_params.fx = fx;
    color_params.fy = fy;
    dev->setColorCameraParams(color_params);
}

/**
 * @brief Get the intrinsic parameters of the RGB camera
 * 
 * @param cx    The x-axis optical center of the camera in pixels.
 * @param cy    The y-axis optical center of the camera in pixels.
 * @param fx    The x-axis focal length of the camera in pixels.
 * @param fy    The y-axis focal length of the camera in pixels.
 */
void KinectDevice::get_color_params(float& cx, float& cy, float& fx, float& fy) {
    cx = color_params.cx;
    cy = color_params.cy;
    fx = color_params.fx;
    fy = color_params.fy;
}

/**
 * @brief Set the intrinsic parameters of the RGB camera
 * 
 * @param cx    The x-axis optical center of the camera in pixels.
 * @param cy    The y-axis optical center of the camera in pixels.
 * @param fx    The x-axis focal length of the camera in pixels.
 * @param fy    The y-axis focal length of the camera in pixels.
 * @param k1    The radial distortion coefficient of the camera: 1st-order.
 * @param k2    The radial distortion coefficient of the camera: 2nd-order.
 * @param k3    The radial distortion coefficient of the camera: 2nd-order.
 * @param p1    The first tangential distortion coefficient of the camera.
 * @param p2    The second tangential distortion coefficient of the camera.
 */
void KinectDevice::set_ir_params(float cx, float cy, float fx, float fy, float k1, float k2, float k3, float p1, float p2) {
    ir_params.cx = cx;
    ir_params.cy = cy;
    ir_params.fx = fx;
    ir_params.fy = fy;
    ir_params.k1 = k1;
    ir_params.k2 = k2;
    ir_params.k3 = k3;
    ir_params.p1 = p1;
    ir_params.p2 = p2;
    dev->setIrCameraParams(ir_params);
}


/**
 * @brief Get the intrinsic parameters of the RGB camera
 * 
 * @param cx    The x-axis optical center of the camera in pixels.
 * @param cy    The y-axis optical center of the camera in pixels.
 * @param fx    The x-axis focal length of the camera in pixels.
 * @param fy    The y-axis focal length of the camera in pixels.
 * @param k1    The radial distortion coefficient of the camera: 1st-order.
 * @param k2    The radial distortion coefficient of the camera: 2nd-order.
 * @param k3    The radial distortion coefficient of the camera: 2nd-order.
 * @param p1    The first tangential distortion coefficient of the camera.
 * @param p2    The second tangential distortion coefficient of the camera.
 */
void KinectDevice::get_ir_params(float& cx, float& cy, float& fx, float& fy, float k1, float k2, float k3, float p1, float p2) {
    cx = ir_params.cx;
    cy = ir_params.cy;
    fx = ir_params.fx;
    fy = ir_params.fy;
    k1 = ir_params.k1;
    k2 = ir_params.k2;
    k3 = ir_params.k3;
    p1 = ir_params.p1;
    p2 = ir_params.p2;
}


/**
 * @brief Initialise the registration for the Kinect v2 using the intrinsic parameters
 * 
 */
void KinectDevice::init_registration() {
    registration = new libfreenect2::Registration(ir_params, color_params);
}


/**
 * @brief Initialise the camera parameters for the Kinect v2
 * 
 */
void KinectDevice::init_params() {
    color_params = dev->getColorCameraParams();
    ir_params = dev->getIrCameraParams();   
}


/**
 * @brief Start the Kinect v2 device
 * 
 * @return The device started successfully
 */
int KinectDevice::start() {
    return dev->start();
    
}


/**
 * 
 * @brief Stop the Kinect v2 device
 * 
 * @return The device stopped successfully
 */
int KinectDevice::stop() {
    return dev->stop();
}

/**
 * @brief Converts small array of depth values (n = numPoints) to points in cartesians space (X,Y,Z)
 * 
 * @param row_arr       Array of rows to be processed.
 * @param col_arr       Array of columns to be processed
 * @param depth_arr     Array of depth values to be processed
 * @param x_arr         Output array of x-values
 * @param y_arr         Output array of x-values
 * @param z_arr         Output array of x-values
 * @param numPoints     Number of points in array to process
 */
void KinectDevice::rowColDepthToXYZ(const float* row_arr, const float* col_arr, const float* depth_arr, float* x_arr, float* y_arr, float* z_arr, int numPoints) {
    
    const float cx = ir_params.cx;
    const float cy = ir_params.cy;
    const float fx = 1 / ir_params.fx;
    const float fy = 1 / ir_params.fy;

    for (int n = 0; n < numPoints; n++) {

        const float row = row_arr[n];
        const float col = col_arr[n];
        const float depth_val = depth_arr[n] / 1000.0f;

        if (!std::isnan(depth_val) && depth_val > 0.001) {

            x_arr[n] = -(col + 0.5 - cx) * fx * depth_val;
            y_arr[n] = (row + 0.5 - cy) * fy * depth_val;
            z_arr[n] = depth_val;
        } else {

            x_arr[n] = 0;
            y_arr[n] = 0;
            z_arr[n] = 0;
        }
    }
}


/**
 * @brief Uses CPU to process depth information to get Point Cloud (ROS - PCL)
 * 
 * @param depth         Depth Image
 * @param registered    Registered Image
 * @param cloud_data    ROS Point Cloud Data
 * @param width         Width of Image
 * @param height        Height of Image
 */
void KinectDevice::getPointCloudCpu(const float* depth, const uint32_t* registered, uint8_t* cloud_data, int width, int height) {

    const float cx = ir_params.cx;
    const float cy = ir_params.cy;
    const float fx = 1 / ir_params.fx;
    const float fy = 1 / ir_params.fy;

    int numElements = width * height;
    const int point_step = 32;
    for (int i = 0; i < numElements; i++) {
        const int row = i / width;
        const int col = i % width;
        const float depth_val = depth[width * row + col] / 1000.0f;
        if (!std::isnan(depth_val) && depth_val > 0.001) {
            uint8_t* ptr = cloud_data + i * point_step;
            /* x-value */
            *(float*)(ptr + 0) = -(col + 0.5 - cx) * fx * depth_val;
            /* y-value */
            *(float*)(ptr + 4) = (row + 0.5 - cy) * fy * depth_val;
            /* z-value */
            *(float*)(ptr + 8) = depth_val;
            /* rgb-value */
            *(uint32_t*)(ptr + 16) = registered[i];
        }
    }
}


/**
 * @brief Wait for the next frame to be received
 * 
 */
void KinectDevice::wait_frames() {
    if (!listener->waitForNewFrame(frames, 10 * 1000)) {
        std::cout << "Error!" << std::endl;
        throw std::exception();
    }
}


/**
 * @brief Return the frame of type (Depth/IR/Color)
 * 
 * @param type The type of frame to be received
 * @return The frame received
 */
libfreenect2::Frame* KinectDevice::get_frame(libfreenect2::Frame::Type type) {
    return frames[type];
}


/**
 * @brief Free the frames from memory
 * 
 */
void KinectDevice::release_frames() {
    listener->release(frames);
}


/**
 * @brief Get the registration object
 * 
 * @return The registration object
 */
libfreenect2::Registration* KinectDevice::get_registration() {
    return registration;
}


/**
 * @brief Destruct the Kinect Device
 * 
 */
KinectDevice::~KinectDevice() {
    delete pipeline;
    delete freenect2;
    delete listener;
    delete registration;
}

}