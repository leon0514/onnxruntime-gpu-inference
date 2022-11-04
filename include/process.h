#ifndef PROCESS_H__
#define PROCESS_H__
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

struct AffineMatrix
{
    float value[6];
};

struct AffineMatrixImage
{
    float value[6];
    float* affine_image;
};

struct Box{
    float left, top, right, bottom, confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

std::vector<Box> gpu_decode(const float* predict, int rows, int cols, float confidence_threshold, float nms_threshold, float* invert_affine_matrix);
void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);

AffineMatrixImage warpaffine(std::string image_path, int width, int height);

#endif