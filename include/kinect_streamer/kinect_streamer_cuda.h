#ifndef KINECT_STREAMER_CUDA_H
#define KINECT_STREAMER_CUDA_H

#include <iostream>

void getPointXYZHelper(const float* D, const uint32_t* R, uint8_t* cloud_data, float cx, float cy, float fx, float fy, int width, int height);

#endif