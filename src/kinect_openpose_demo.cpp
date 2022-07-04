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
        cv::resizeWindow(serial, cv::Size(1280, 720));
    }
    
    
    op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};

    const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String("-1x160"), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String("/home/medrobotics/openpose/models/"),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapper.configure(wrapperStructPose);
    
    opWrapper.start();


    signal(SIGINT, my_handler);
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
    
            cv::Mat img_bgr;
            cv::cvtColor(img_color, img_bgr, cv::COLOR_BGRA2BGR);
            const cv::Mat cvImageToProcess = img_bgr.clone();
            const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
            auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
            const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumProcessed->at(0)->cvOutputData);
            auto poseKeypoints = datumProcessed->at(0)->poseKeypoints;
            const auto numberPeopleDetected = poseKeypoints.getSize(0);
            const auto numberBodyParts = poseKeypoints.getSize(1);
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
                        cv::circle(img_bgr, cv::Point(col, row), 12, cv::Scalar(0, 0, 255), -1);
                        col_arr[numPoints] = col;
                        row_arr[numPoints] = row;
                        depth_arr[numPoints] = depth->data[depth->width * (int)row + (int)col];
                        numPoints++;
                    }
                }


                float* x_arr = (float*)malloc(sizeof(float) * numPoints);
                float* y_arr = (float*)malloc(sizeof(float) * numPoints);
                float* z_arr = (float*)malloc(sizeof(float) * numPoints);

                kin_devs[serial]->rowColDepthToXYZ(row_arr, col_arr, depth_arr, x_arr, y_arr, z_arr, numPoints);
                if (numPoints > 0) {
                    for (int i = 0; i < numPoints; i++) {
                        if (x_arr[i] == 0 && y_arr[i] == 0 && z_arr[i] == 0) {
                            continue;
                        }
                    }
                }
                
                free(row_arr);
                free(col_arr);
                free(depth_arr);

                free(x_arr);
                free(y_arr);
                free(z_arr);
            }
            cv::flip(img_bgr, img_bgr, 1);
            cv::imshow(serial, img_bgr);
            cv::waitKey(1);
            
            kin_devs[serial]->release_frames();
        }
    }

    for (std::string serial : serials) {
        kin_devs[serial]->stop();
    }
}

