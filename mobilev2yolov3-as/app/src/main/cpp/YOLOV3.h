//
// Created by root on 6/3/19.
//

#ifndef YOLOV3_H
#define YOLOV3_H


#include <iostream>
#include <vector>
using namespace std;

#include "mobilenet_v01.id.h"
#include "net.h"




class YOLOV3 {

public:
    YOLOV3(string param_path, string bin_path);
	YOLOV3(ncnn::Mat param_path, ncnn::Mat bin_path);

	ncnn::Mat detect(ncnn::Mat in);

    ~YOLOV3();

private:

    ncnn::Net yolov3_net;

    const int target_size = 320;
	const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
	const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};

};


#endif //YOLOV3_H
