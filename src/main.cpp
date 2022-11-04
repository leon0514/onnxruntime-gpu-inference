#include "model.h"
#include <stdio.h>
#include <sys/time.h>


using namespace std;
using namespace cv;


// test cost time
/*
class Time
{
public:
    Time(){}
    ~Time(){}
 
    void start()
    {
        gettimeofday(&tv1,nullptr);
    }
 
    int cost()
    {
        gettimeofday(&tv2,nullptr);
        return (1000000*(tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec));
    }
private:
    struct timeval tv1, tv2;
};
*/


int main()
{
    NetConfig cfg = {
        0.5,
        0.45,
        "onnx_model/yolov5.onnx",
    };
    Yolov5 model(cfg, 0);
    string image_path = "images/3.jpg";

    for (int i = 0; i < 10; i++)
    {
        auto boxes = model.detect(image_path);
        cout << "box length:"<<boxes.size() << endl;
        // only draw box
        // model.save_detect_image(image_path, boxes);
    }
    return 0;
}