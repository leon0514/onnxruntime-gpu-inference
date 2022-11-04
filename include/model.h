#ifndef MODEL_H__
#define MODEL_H__

#include <string>
#include <vector>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "onnxruntime_cxx_api.h"
#include "cuda_provider_factory.h"
#include "check.h"
#include "process.h"

typedef struct NetConfig{
    float nms_thresh;
    float conf_thresh;
    std::string model_path;
    std::string class_name_path;
} NetConfig;



class Yolov5{
public:
    Yolov5(NetConfig cfg, int device_id);
    ~Yolov5();

public:
    std::vector<Box> detect(std::string image_path);
    void save_detect_image(std::string image_path, std::vector<Box> result);

private:
    int m_width;
    int m_height;

    float m_nms_thresh;
    float m_conf_thresh;

    Ort::Env m_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "YOLOv5");
	Ort::Session *m_ort_session = nullptr;
	Ort::SessionOptions m_sessionOptions = Ort::SessionOptions();
	std::vector<char*> m_input_names;
	std::vector<char*> m_output_names;
	std::vector<std::vector<int64_t>> m_input_node_dims; // >=1 outputs
	std::vector<std::vector<int64_t>> m_output_node_dims; // >=1 outputs

private:
    void load_model(std::string model_path, int device_id);
    void postprocess(std::vector<Box> boxes);

};

#endif