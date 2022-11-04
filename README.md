# onnxruntime-gpu-inference
使用gpu推理onnx yolov5模型，同时将前处理和后处理都是用gpu处理

## 感谢杜老
1. cuda warpaffine前处理代码是按照杜老b站上的教学一点点敲的
2. cuda yolov5后处理是从 https://github.com/shouxieai/learning-cuda-trt 搬过来的
3. Makefile 也是和杜老一点点学习的，学会之后，几乎所有的Makefile都是这个模板了
 

## 一些细节
在cuda中实现预处理， 下面几句代码将RGBRGBRGB排列的数据改为RRR...GGG...BBB...排列， 并做归一化处理， 除以255
```C++
float* pdst = output_image + dy * output_image_width + dx;
// RRR...GGG...BBB.. 255
// 归一化
pdst[0] = c2 / 255.0; 
pdst[output_image_width*output_image_height] = c1 / 255.0; 
pdst[2*output_image_width*output_image_height] = c0 / 255.0;
```
除以255之后得到的是浮点值，output_image需要时float指针
