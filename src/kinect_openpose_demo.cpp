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

#include <fstream>
#include <ctime>

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
        kin_devs[serial] = kin_dev;
    }

    for (std::string serial : serials) {
        // For each device, initialise the intrinsic parameters from the device
        kin_devs[serial]->init_params();
        // For each device, initialise the registration object
        kin_devs[serial]->init_registration();
    }

    for (std::string serial : serials) {
        cv::namedWindow(serial, cv::WINDOW_NORMAL);
    }
    
    
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};


    const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
    const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x160");
    const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
    const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
    const auto poseMode = op::flagsToPoseMode(FLAGS_body);
    const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
    
    const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
    const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
    const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
    const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
    const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
    const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
    const bool enableGoogleLogging = true;

    const op::WrapperStructPose wrapperStructPose{
        poseMode,
        netInputSize,
        FLAGS_net_resolution_dynamic,
        outputSize,
        keypointScaleMode,
        FLAGS_num_gpu,
        FLAGS_num_gpu_start,
        FLAGS_scale_number,
        (float)FLAGS_scale_gap,
        op::flagsToRenderMode(FLAGS_render_pose, multipleView),
        poseModel,
        !FLAGS_disable_blending,
        (float)FLAGS_alpha_pose,
        (float)FLAGS_alpha_heatmap,
        FLAGS_part_to_show,
        op::String("/home/medrobotics/openpose/models"),
        heatMapTypes, heatMapScaleMode,
        FLAGS_part_candidates,
        (float)FLAGS_render_threshold,
        FLAGS_number_people_max,
        FLAGS_maximize_positives,
        FLAGS_fps_max,
        op::String(FLAGS_prototxt_path),
        op::String(FLAGS_caffemodel_path),
        (float)FLAGS_upsampling_ratio,
        enableGoogleLogging
    };
    
    opWrapper.configure(wrapperStructPose);
    opWrapper.start();
    std::ofstream of("data.csv", std::ios::trunc);
    signal(SIGINT, my_handler);
    int no = 0;
    float x_prev = 0;
    float y_prev = 0;
    float z_prev = 0;
    clock_t current_ticks, delta_ticks;
    clock_t fps = 0;
    current_ticks = clock();
    while (!flag) {
        for (std::string serial : serials) {
            kin_devs[serial]->KinectDevice::wait_frames();

            libfreenect2::Frame* color = kin_devs[serial]->get_frame(libfreenect2::Frame::Color);
            libfreenect2::Frame* depth = kin_devs[serial]->get_frame(libfreenect2::Frame::Depth);

            cv::Mat img_color(cv::Size(color->width, color->height), CV_8UC4, color->data);
            cv::Mat img_depth(cv::Size(depth->width, depth->height), CV_32FC1, depth->data);


            libfreenect2::Registration* registration = kin_devs[serial]->get_registration();

            std::unique_ptr<libfreenect2::Frame> undistorted = std::make_unique<libfreenect2::Frame>(512, 424, 4);
            std::unique_ptr<libfreenect2::Frame> registered = std::make_unique<libfreenect2::Frame>(512, 424, 4);

            cv::Mat img_undistorted(cv::Size(undistorted->width, undistorted->height), CV_32FC1, undistorted->data);
            cv::Mat img_registered(cv::Size(registered->width, registered->height), CV_8UC4, registered->data);

            registration->apply(color, depth, undistorted.get(), registered.get());
            registration->undistortDepth(depth, undistorted.get());


                                delta_ticks = clock() - current_ticks;
                                fps = CLOCKS_PER_SEC / delta_ticks;
                                std::cout << fps << std::endl;
                                current_ticks = clock();

    
            cv::Mat img_bgr;
            cv::cvtColor(img_color, img_bgr, cv::COLOR_BGRA2BGR);
            cv::cvtColor(img_registered, img_registered, cv::COLOR_BGRA2BGR);
            
            cv::Mat img_undistorted_bgr;
            img_undistorted.copyTo(img_undistorted_bgr);
            img_undistorted_bgr /= 255.0;
            img_undistorted_bgr /= 20.0;
            cv::cvtColor(img_undistorted_bgr, img_undistorted_bgr, cv::COLOR_GRAY2BGR);


            cv::Mat img_thresh1;
            cv::Mat img_thresh2;
            cv::Mat img_thresh;
            
            cv::threshold(img_depth, img_thresh1, 1500, 65535, cv::THRESH_BINARY_INV);
            cv::threshold(img_depth, img_thresh2, 1, 65535, cv::THRESH_BINARY);
            cv::bitwise_and(img_thresh1, img_thresh2, img_thresh);
            cv::Mat mask;
            img_thresh.convertTo(mask, CV_8UC1);

            cv::Mat img_seg;
            cv::copyTo(img_registered, img_seg, mask);
            
            const cv::Mat cvImageToProcess = img_seg;

            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
            auto poseKeypoints = datumProcessed->at(0)->poseKeypoints;
            const auto numberPeopleDetected = poseKeypoints.getSize(0);
            const auto numberBodyParts = poseKeypoints.getSize(1);
            if (0) {
                if (numberPeopleDetected > 0 && numberBodyParts > 0) {
                    float* row_arr = (float*)malloc(sizeof(float) * numberBodyParts * numberPeopleDetected);
                    float* col_arr = (float*)malloc(sizeof(float) * numberBodyParts * numberPeopleDetected);
                    float* depth_arr = (float*)malloc(sizeof(float) * numberBodyParts * numberPeopleDetected);
                    int numPoints = 0;
                    for (int n = 0; n < numberPeopleDetected; n++) {
                        for (int m = 0; m < numberBodyParts; m++) {
                            const double col = poseKeypoints[{n, m, 0}];
                            const double row = poseKeypoints[{n, m, 1}];
                            const double con = poseKeypoints[{n, m, 2}];
                            col_arr[numPoints] = col;
                            row_arr[numPoints] = row;
                            depth_arr[numPoints] = undistorted->data[depth->width * (int)row + (int)col];
                            if (n == 0 && m  == 4) {
                                cv::circle(img_seg, cv::Point(col, row), 3, cv::Scalar(0, 0, 255), -1);
                                cv::circle(img_undistorted_bgr, cv::Point(col, row), 3, cv::Scalar(0, 0, 255), -1);
                                float x;
                                float y;
                                float z;

                                kin_devs[serial]->rowColDepthToXYZ_scalar(row, col, undistorted->data[depth->width * (int)row + (int)col], x, y, z);
                                float vel2 = (x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev) + (z - z_prev) * (z - z_prev);
                                if (depth_arr[numPoints] != 0 && con >= 0.35 && !(x == 0 && y == 0 && z == 0) && (z > 0.025) && (vel2 < 0.25 * 0.25)) {
                                    of << x << "," << y << "," << z << std::endl;
                                } else {
                                    /*
                                    if (depth_arr[numPoints] == 0) {
                                        std::cout << "Depth invalid." << std::endl;
                                    }
                                    if (con < 0.35) {
                                        std::cout << "Confidence too low" << std::endl;
                                    }
                                    if (x == 0 && y == 0 && z == 0) {
                                        std::cout << "Coordinates invalid (0, 0, 0)" << std::endl;
                                    }
                                    if (z <= 0.35) {
                                        std::cout << "Z-coordinate too close" << std::endl;
                                        std::cout << "\t- " << z << std::endl;
                                    }
                                    if (vel2 >= 0.25 * 0.25) {
                                        std::cout << "Velocity is too fast" << std::endl;
                                    }*/
                                }
                                x_prev = x;
                                y_prev = y;
                                z_prev = z;
                            }

                            numPoints++;
                        }
                        break;
                    }

                    /*
                    float* x_arr = (float*)malloc(sizeof(float) * numPoints);
                    float* y_arr = (float*)malloc(sizeof(float) * numPoints);
                    float* z_arr = (float*)malloc(sizeof(float) * numPoints);
                    */

                    /*
                    free(row_arr);
                    free(col_arr);
                    free(depth_arr);

                    free(x_arr);
                    free(y_arr);
                    free(z_arr);
                    */
                }
            }
            cv::flip(img_bgr, img_bgr, 1);
            cv::imshow(serial, img_undistorted_bgr);
            cv::imshow(serial + "_openpose", cvMat);
            cv::waitKey(1);
            
            kin_devs[serial]->release_frames();
        }
    }
    of.close();

    for (std::string serial : serials) {
        kin_devs[serial]->stop();
    }
}

