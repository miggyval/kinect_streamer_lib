#include <iostream>
#include <fstream>
#include <string>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <chrono>   
#include <unistd.h>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/core/cuda_types.hpp>

#include <kinect_streamer/kinect_streamer.hpp>


#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

int flag = false;

void pre_handler(int s) {
    std::cout << "Quitting!" << "\n\r";
    exit(-1);
}

void my_handler(int s) {
    std::cout << "Quitting!" << "\n\r";
    flag = true;
}


int main(int argc, char** argv) {

    signal(SIGINT, pre_handler);

    libfreenect2::setGlobalLogger(NULL);
    libfreenect2::Freenect2 freenect2;
    std::map<std::string, KinectStreamer::KinectDevice*> kin_devs;
    std::vector<std::string> serials;
    int num_devices = freenect2.enumerateDevices();
    if (num_devices == 0) {
        std::cout << "No devices detected!" << "\n\r";
        exit(-1);
    } else {
        std::cout << "Connected devices:" << "\n\r";
        for (int idx = 0; idx < num_devices; idx++) {
            std::cout << "- " << freenect2.getDeviceSerialNumber(idx) << "\n\r";
            serials.push_back(freenect2.getDeviceSerialNumber(idx));
        }
    }

    int n = serials.size();

    for (std::string serial : serials) {
        KinectStreamer::KinectDevice* kin_dev = new KinectStreamer::KinectDevice(serial);
        if (!kin_dev->start()) {
            std::cout << "Failed to start Kinect Serial no.: " << serial << std::endl;
            exit(-1);
        }
        kin_dev->init_registration();
        kin_devs[serial] = kin_dev;
    }


    for (std::string serial : serials) {
        cv::namedWindow(serial, cv::WINDOW_NORMAL);
        cv::resizeWindow(serial, cv::Size(1280, 720));
    }
    
    
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
    
    opWrapper.start();


    signal(SIGINT, my_handler);
    while (!flag) {
        for (std::string serial : serials) {
            kin_devs[serial]->KinectDevice::wait_frames();

            libfreenect2::Frame* color = kin_devs[serial]->get_frame(libfreenect2::Frame::Color);
            libfreenect2::Frame* depth = kin_devs[serial]->get_frame(libfreenect2::Frame::Depth);

            cv::Mat img_color(cv::Size(color->width, color->height), CV_8UC4, color->data);
            cv::Mat img_depth(cv::Size(depth->width, depth->height), CV_32FC1, depth->data);
        

            cv::Mat img_bgr;
            cv::cvtColor(img_color, img_bgr, cv::COLOR_BGRA2BGR);
            const cv::Mat cvImageToProcess = img_bgr.clone();
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
            auto poseKeypoints = datumProcessed->at(0)->poseKeypoints;
            const auto numberPeopleDetected = poseKeypoints.getSize(0);
            const auto numberBodyParts = poseKeypoints.getSize(1);
            for (int n = 0; n < numberPeopleDetected; n++) {
                for (int m = 0; m < numberBodyParts; m++) {
                    const double x = poseKeypoints[{n, m, 0}];
                    const double y = poseKeypoints[{n, m, 1}];
                    const double c = poseKeypoints[{n, m, 2}];
                    cv::circle(img_bgr, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
                }
            }
            cv::flip(img_bgr, img_bgr, 1);
            cv::imshow(serial, img_bgr);
            cv::waitKey(1);
            
            kin_devs[serial]->release_frames();
        }
    }
}

