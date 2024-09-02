#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#define IMG_DIR "./img_test/test_000.png"
#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224
#define CLS 5
using namespace std;
using namespace cv;
const auto size = cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT);
uint8_t colorB[] = {0, 255, 69, 0, 255};
uint8_t colorG[] = {0, 0, 47, 0, 255};
uint8_t colorR[] = {0, 0, 142, 255, 0};

//void task(float *result){  
int main(void){
    float *outTensor = new float[IMAGE_WIDTH * IMAGE_HEIGHT];
    cv::Mat image;
    cv::Mat input_image = cv::imread(IMG_DIR, IMREAD_UNCHANGED);
    Mat segMat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_8UC3);
    //cv::resize(input_image, image, size, cv::INTER_NEAREST);    
    input_image.convertTo(image, CV_8U);
    for (int row = 0; row < IMAGE_WIDTH; row++) {
       for (int col = 0; col < IMAGE_HEIGHT; col++) {
           //result[row*IMAGE_HEIGHT + col] = image.at<Vec3d>(row, col);  
           //cout << "B: " << static_cast<int>(image.at<uchar>(row, col))  << endl;        
           outTensor[row* IMAGE_HEIGHT + col] = image.at<uchar>(row, col); 
       }
    }
    for (int row = 0; row < IMAGE_WIDTH; row++) {
       for (int col = 0; col < IMAGE_HEIGHT; col++) {
          int pos = outTensor[row* IMAGE_HEIGHT + col];
          //cout << "B: " << posit << endl; 
          segMat.at<Vec3b>(row, col) = Vec3b(colorB[pos], colorG[pos], colorR[pos]);
          //cout << "B: " << segMat.size()  << endl;
       }
    }
    cv::imwrite("output/seg.jpg", segMat);
    delete[] outTensor;
}

