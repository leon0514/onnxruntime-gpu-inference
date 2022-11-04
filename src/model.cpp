#include <fstream>
#include <sstream>
#include <cuda_runtime.h>
#include "model.h"


using namespace std;
using namespace cv;
using namespace Ort;


Yolov5::Yolov5(NetConfig cfg, int device_id)
{
    m_nms_thresh = cfg.nms_thresh;
    m_conf_thresh = cfg.conf_thresh;
    string model_path = cfg.model_path;

    // load model
    load_model(model_path, device_id);
}

Yolov5::~Yolov5()
{
    delete m_ort_session;
}

void Yolov5::load_model(string model_path, int device_id)
{
    // Sets graph optimization level
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    std::cout << "Sets graph optimization level ORT_ENABLE_BASIC" << std::endl;
    m_sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // using gpu 0
    std::cout << "Using gpu id " << device_id << std::endl;
    OrtSessionOptionsAppendExecutionProvider_CUDA(m_sessionOptions, device_id);

    // init model
    std::cout << "Initialize model : " << model_path << std::endl;
    m_ort_session = new Session(m_env, model_path.c_str(), m_sessionOptions);
    size_t numInputNodes = m_ort_session->GetInputCount();
    size_t numOutputNodes = m_ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;

    // get input
    std::cout << "Get input,  numInputNodes is " << numInputNodes << std::endl;
    for (int i = 0; i < numInputNodes; ++i)
    {
        m_input_names.emplace_back(m_ort_session->GetInputName(i, allocator));
        std::cout << "input_names " << i << " is " << m_input_names[i] << std::endl;
        TypeInfo input_type_info = m_ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        m_input_node_dims.emplace_back(input_dims);
    }
    
    // get output
    std::cout << "Get output numOutputNodes is " << numOutputNodes << std::endl;
    for(int i = 0; i < numOutputNodes; ++i)
    {
        m_output_names.emplace_back(m_ort_session->GetOutputName(i, allocator));
        std::cout << "output_names " << i << " is " << m_output_names[i] << numInputNodes << std::endl;
        TypeInfo output_type_info = m_ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        m_output_node_dims.emplace_back(output_dims);
    }

    // get input image width and height
    std::cout << "Get input shape" << std::endl;
    m_width = m_input_node_dims[0][2];
    m_height = m_input_node_dims[0][3];
    std::cout << "Batch : " <<  m_input_node_dims[0][0]  << ", "
              << "Channel : " <<  m_input_node_dims[0][1]  << ", "
              << "Width : " << m_width << ", "
              << "Height : " << m_height << std::endl;
}


vector<Box> Yolov5::detect(string image_path)
{
    // auto blob = preprocess(image_path);
    AffineMatrixImage afi = warpaffine(image_path, m_width, m_height);
    float* blob = afi.affine_image;
    float value[6];
    memcpy(value, afi.value, 6*sizeof(float));
    // cout << blob.size() << endl;
    array<int64_t, 4> input_shape_{ 1, 3, m_height, m_width };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, blob,  m_width*m_height*3, input_shape_.data(), input_shape_.size());
    free(blob);
	vector<Value> ort_outputs = m_ort_session->Run(RunOptions{ nullptr }, &m_input_names[0], &input_tensor_, 1, m_output_names.data(), m_output_names.size());

	auto output_dims = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	int num_proposal = output_dims[1]; // 25200

	int nout = output_dims[2]; // 6

    
    const float* pbox = ort_outputs[0].GetTensorData<float>();

    // cout << num_proposal << "    " << nout << endl;
    auto boxes = gpu_decode(pbox, num_proposal, nout, m_conf_thresh, m_nms_thresh, value);
    return boxes;
}


void Yolov5::save_detect_image(string image_path, vector<Box> result)
{
    Mat image = imread(image_path);
    for (auto box : result)
    {
        int xmin = (int)box.left;
        int ymin = (int)box.top;
        int xmax = (int)box.right;
        int ymax = (int)box.bottom;
        rectangle(image, cvPoint(xmin, ymin), cvPoint(xmax, ymax), cvScalar(0, 255, 0), 3, 4, 0 );
    }
    imwrite("result.jpg", image);
}

