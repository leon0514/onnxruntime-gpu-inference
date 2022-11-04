#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include "process.h"
#include "check.h"

using namespace std;
using namespace cv;


/*
  输出图某一点坐标，乘 仿射变换矩阵的逆矩阵，得到对应原图坐标
*/
__device__ void affine_proj(float* matrix, int x, int y, float* proj_x, float* proj_y)
{
    // matrix
    // m0, m1, m2
    // m3, m4, m5

    //            m0, m1, m2
    //(x, y ,1) * m3, m4, m5 = (proj_x, proj_y, _)
    //            0,  0,  1
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warpaffine_kernel(
    uint8_t *input_image, int input_image_width, int input_image_height,
    float *output_image, int output_image_width, int output_image_height,
    AffineMatrix m, uint8_t const_value
)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= output_image_width || dy >= output_image_height) return;
    // 先将填充值设置为默认
    float c0 = const_value;
    float c1 = const_value;
    float c2 = const_value;

    // 接收原图坐标
    float src_x = 0;
    float src_y = 0;
    affine_proj(m.value, dx, dy, &src_x, &src_y);

    /* 双线性插值*/
    if (
        src_x < -1 || 
        src_x >= input_image_width || 
        src_y < -1 || 
        src_y >= input_image_height
    )
    {
        // 对应原图坐标超出界限
    }
    else
    {
        /* 
            -----双线性插值-----

            o 表示当前点 变换图对应原图的点
            A, B, C, D 表示该点周围的四个像素点

            插值权重
            A (x_low, y_low)         B (x_high, y_low)
            --------------------------
            |        |               |
            |   w0   |      w1       |
            ---------o----------------
            |        |(src_x, src_y) |
            |        |               |
            |   w3   |      w2       |
            |        |               |
            --------------------------
            D (x_low, y_high)        C (x_high, y_high)

            left   = src_x - x_low
            top    = src_y - y_low
            right  = x_high - src_x
            bottom = y_high - src_y

            计算面积，即权重

            w0 = left   * top
            w1 = top    * right
            w2 = right  * bottom
            w3 = bottom * left

            再将 o 的值赋值给变换后的坐标位置的RGB的值

            o[0] = A[0] * w3 + B[0] * w2 + C[0] * w0 + D[0] * w1
            o[1] = A[1] * w3 + B[1] * w2 + C[1] * w0 + D[1] * w1 
            o[2] = A[2] * w3 + B[2] * w2 + C[2] * w0 + D[2] * w1
        */
        int x_low  = floor(src_x);
        int y_low  = floor(src_y);
        int x_high = x_low + 1;
        int y_high = y_low + 1;

        uint8_t const_value_array[] = {
            const_value, const_value, const_value
        };
        float left   = src_x  - x_low;
        float top    = src_y  - y_low;
        float right  = x_high - src_x;
        float bottom = y_high - src_y;
        float w0     = left   * top;
        float w1     = top    * right;
        float w2     = right  * bottom;
        float w3     = bottom * left;

        // 四个点的像素值先设置为默认
        uint8_t* v0 = const_value_array;
        uint8_t* v1 = const_value_array;
        uint8_t* v2 = const_value_array;
        uint8_t* v3 = const_value_array;

        /* 
            像素排列
            RGBRGBRGB
            RGBRGBRGB
            RGBRGBRGB
        */  
        
        if (x_low >=0 && y_low >=0 && x_low < input_image_width && y_low < input_image_height)
        {
            v0 = input_image + (y_low * input_image_width + x_low) * 3;
        }
        if (x_high >=0 && y_low >=0 && x_high < input_image_width && y_low < input_image_height)
        {
            v1 = input_image + (y_low * input_image_width + x_high) * 3;
        }
        if (x_high >=0 && y_high >=0 && x_high < input_image_width && y_high < input_image_height)
        {
            v2 = input_image + (y_high * input_image_width + x_high) * 3;
        }
        if (x_low >=0 && y_high >=0 && x_low < input_image_width && y_high < input_image_height)
        {
            v3 = input_image + (y_high * input_image_width + x_low) * 3;
        }
        c0 = floorf(w0 * v2[0] + w1 * v3[0] + w2 * v0[0] + w3 * v1[0] + 0.5f);
        c1 = floorf(w0 * v1[1] + w1 * v2[1] + w2 * v0[1] + w3 * v1[1] + 0.5f);
        c2 = floorf(w0 * v1[2] + w1 * v2[2] + w2 * v0[2] + w3 * v1[2] + 0.5f);
    }
    // assign
    // float* pdst = output_image + (dy * output_image_width + dx) * 3;
    // pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;
    // BGR--->RGB /255 
    // pdst[2] = c0 / 255.0; pdst[1] = c1 / 255.0; pdst[0] = c2 / 255.0;

    
    float* pdst = output_image + dy * output_image_width + dx;
    // RRR...GGG...BBB...排列
    // 归一化
    pdst[0] = c2 / 255.0; 
    pdst[output_image_width*output_image_height] = c1 / 255.0; 
    pdst[2*output_image_width*output_image_height] = c0 / 255.0;
    
}

Mat get_affine_matrix(const Size &input, const Size &output)
{
    float scale_x = output.width / (float)input.width;
    float scale_y = output.height / (float)input.height;
    float scale_factor = min(scale_x, scale_y);
    Mat scale_matrix = (Mat_<float>(3,3) <<
        scale_factor, 0,           0,
        0,            scale_factor, 0,
        0,            0,           1
    );
    Mat translation_matrix = (Mat_<float>(3,3) <<
        1, 0, -input.width * 0.5 * scale_factor + output.width * 0.5,
        0, 1, -input.height * 0.5 * scale_factor + output.height * 0.5,
        0, 0, 1
    );
    Mat affine_matrix = translation_matrix * scale_matrix;
    affine_matrix = affine_matrix(Rect(0, 0, 3, 2));

    return affine_matrix;
}

AffineMatrix get_gpu_affine_matrix(const Size &input, const Size &output)
{
    auto affine_matrix = get_affine_matrix(input, output);
    Mat invert_affine_matrix;
    // 求逆矩阵
    cv::invertAffineTransform(affine_matrix, invert_affine_matrix);

    AffineMatrix am;
    memcpy(am.value, invert_affine_matrix.ptr<float>(0), sizeof(am.value));
    return am;
}

AffineMatrixImage warpaffine(std::string image_path, int width, int height)
{
    Mat image = imread(image_path);
    size_t image_bytes = image.rows * image.cols * 3;
    uint8_t *image_deivce = nullptr;

    // Mat affine(width, height, CV_32FC3);
    float* affine;
    affine = (float*)malloc(width*height*3*sizeof(float));
    
    // size_t affine_bytes = affine.rows * affine.cols * 3 * sizeof(float);
    size_t affine_bytes = 640*640*3 * sizeof(float);
    float *affine_device = nullptr;

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    checkRuntime(cudaMalloc(&image_deivce, image_bytes));
    checkRuntime(cudaMalloc(&affine_device, affine_bytes));
    checkRuntime(cudaMemcpy(image_deivce, image.data, image_bytes, cudaMemcpyHostToDevice));
    // auto gpu_M = get_gpu_affine_matrix(image.size(), affine.size());
    auto gpu_M = get_gpu_affine_matrix(image.size(), Size(height, width));

    dim3 block_size(32, 32); // blocksize最大就是1024，这里用2d来看更好理解
    // dim3 grid_size((affine.cols + 31) / 32, (affine.rows + 31) / 32);
    dim3 grid_size((width + 31) / 32, (height + 31) / 32);

    /*
    grid:20*20   block:32*32

    GRID
    ------------------
    |  block |       |
    ------------------
    |        |    o  |
    ------------------ 

    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;

    id = dy * gridDim.x * blockDim.x + dx

    */

    // warpaffine_kernel<<<grid_size, block_size, 0, stream>>>(
    //     image_deivce,  
    //     image.cols, 
    //     image.rows,
    //     affine_device, 
    //     affine.cols,
    //     affine.rows,
    //     gpu_M,
    //     114
    // );
    warpaffine_kernel<<<grid_size, block_size, 0, stream>>>(
        image_deivce,  
        image.cols, 
        image.rows,
        affine_device, 
        width,
        height,
        gpu_M,
        114
    );
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(affine, affine_device, affine_bytes, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(image_deivce));
    checkRuntime(cudaFree(affine_device));
    // imwrite("warpaffine-gpu.jpg", affine);
    AffineMatrixImage afi;
    memcpy(afi.value, gpu_M.value, 6*sizeof(float));
    afi.affine_image = affine;
    return afi;
}