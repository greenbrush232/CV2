#include <iostream>
#include <opencv2/opencv.hpp>
#include "Header.h"

using namespace std;

void main()
{
    string image_path = "balls.jpg";
    cv::Mat source_image = cv::imread(image_path);
    cout << "Initial params: " << source_image.cols << ", " << source_image.rows << ", " << source_image.channels() << endl;

    auto filters = getDistribution(0, 1);
    cv::Mat convolutionLayer = Convolution(source_image, 1, filters);
    cout << "After convolution: " << convolutionLayer.cols << ", " << convolutionLayer.rows << ", " << convolutionLayer.channels() << endl;
    cv::Mat normalizeLayer = Normalize(convolutionLayer, 1, 1);
    cout << "After norm: " << normalizeLayer.cols << ", " << normalizeLayer.rows << ", " << normalizeLayer.channels() << endl;
    cv::Mat reluLayer = Relu(normalizeLayer);
    cout << "After ReLU: " << reluLayer.cols << ", " << reluLayer.rows << ", " << reluLayer.channels() << endl;
    cv::Mat maxPoolingLayer = MaxPooling(reluLayer, 2, 2);
    cout << "After max pooling: " << maxPoolingLayer.cols << ", " << maxPoolingLayer.rows << ", " << maxPoolingLayer.channels() << endl;
    cv::Mat l5 = Softmax(maxPoolingLayer);

    cv::imshow("Image", source_image);
    cv::waitKey(0);

    system("pause");
}

