#include "check.h"
#include "process.h"

using namespace std;


vector<Box> gpu_decode(const float* predict, 
                       int rows, 
                       int cols, 
                       float confidence_threshold, 
                       float nms_threshold, 
                       float* invert_affine_matrix)
{
    
    vector<Box> box_result;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float* predict_device = nullptr;
    float* output_device = nullptr;
    float* output_host = nullptr;
    float* invert_affine_matrix_device = nullptr;
    int max_objects = 1000;
    int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    checkRuntime(cudaMalloc(&predict_device, rows * cols * sizeof(float)));
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    checkRuntime(cudaMalloc(&invert_affine_matrix_device, 6 * sizeof(float)));

    checkRuntime(cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    checkRuntime(cudaMemcpyAsync(invert_affine_matrix_device, invert_affine_matrix, 6 * sizeof(float), cudaMemcpyHostToDevice, stream));
    decode_kernel_invoker(
        predict_device, rows, cols - 5, confidence_threshold, 
        nms_threshold, invert_affine_matrix_device, output_device, max_objects, NUM_BOX_ELEMENT, stream
    );
    checkRuntime(cudaMemcpyAsync(output_host, output_device, 
        sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, stream
    ));
    checkRuntime(cudaStreamSynchronize(stream));

    int num_boxes = min((int)output_host[0], max_objects);
    for(int i = 0; i < num_boxes; ++i){
        float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if(keep_flag){
            box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]
            );
        }
    }
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFree(predict_device));
    checkRuntime(cudaFree(output_device));
    checkRuntime(cudaFree(invert_affine_matrix_device));
    checkRuntime(cudaFreeHost(output_host));
    return box_result;
}



