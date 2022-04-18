#pragma once
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
typedef cv::Vec<float, 5> Vec5f;

vector<vector<vector<vector<double>>>> getDistribution(double x_mu, double x_sigma)
{
    mt19937 gen((std::random_device())());
    normal_distribution<double> nd(x_mu, x_sigma);
    vector<vector<vector<vector<double>>>> result;
    for (int i = 0; i < 3; i++) {
        result.push_back(vector<vector<vector<double>>>());
        for (int j = 0; j < 3; j++) {
            result[i].push_back(vector<vector<double>>());
            for (int k = 0; k < 3; k++) {
                result[i][j].push_back(vector<double>());
                for (int l = 0; l < 5; l++) {
                    result[i][j][k].push_back(nd(gen));
                }
            }
        }
    }

    return result;
}

cv::Mat Convolution(cv::Mat img, int step, vector<vector<vector<vector<double>>>> filters) {
    int src_height = img.rows;
    int src_width = img.cols;
    int f_width = filters.size();
    int f_height = filters[0].size();
    int f_channels = filters[0][0].size();
    int f_count = filters[0][0][0].size();
    int res_width = int((src_height - f_width) / step + 1);
    int res_height = int((src_height - f_height) / step + 1);
    cv::Mat res_layer = cv::Mat(cv::Size(res_width, res_height), CV_32FC(f_count));

    for (int i = 0; i < f_count; i++) {
        for (int j = 0; j < res_height; j++) {
            for (int k = 0; k < res_width; k++) {
                double resVal = 0;
                for (int n = 0; n < f_height; n++) {
                    for (int m = 0; m < f_width; m++) {
                        for (int l = 0; l < f_channels; l++) {
                            resVal += img.at<cv::Vec3b>(k * step + m, j * step + n)[l] * filters[n][m][l][i];
                        }
                    }
                }
                res_layer.at<Vec5f>(j, k)[i] = resVal;
            }
        }
    }

    return res_layer;
}

cv::Mat Normalize(cv::Mat img, int gamma, float beta) {
    vector<double> std;
    vector<double> mean;
    cv::meanStdDev(img, mean, std);
    double root = sqrt(std[0] * std[0]);
    cv::Mat tmp2;
    cv::subtract(img, mean[0], tmp2);
    cv::Mat div;
    cv::divide(tmp2, root, div);
    div = div.mul(gamma);

    return div;
}

cv::Mat Relu(cv::Mat img) {
    for (int i = 0; i < img.cols; i++) {
        for (int j = 0; j < img.rows; j++) {
            for (int k = 0; k < 5; k++) {
                if (img.at<Vec5f>(i, j)[k] < 0) {
                    img.at<Vec5f>(i, j)[k] = 0;
                }
            }
        }
    }

    return img;
}

cv::Mat MaxPooling(cv::Mat img, int height, int width) {
    int src_height = img.rows;
    int src_width = img.cols;
    int src_count = 5;
    int res_width = int(src_height / height);
    int res_height = int(src_width / width);
    cv::Mat res_layer = cv::Mat(cv::Size(res_width, res_height), CV_32FC(src_count));

    for (int i = 0; i < res_height; i++) {
        for (int j = 0; j < res_width; j++) {
            Vec5f resVal = Vec5f(0, 0, 0, 0, 0);
            for (int n = height * i; n < height * (i + 1); n++) {
                for (int m = width * j; m < width * (j + 1); m++) {
                    for (int k = 0; k < 5; k++) {
                        if (img.at<Vec5f>(m, n)[k] > resVal[k])
                            resVal[k] = img.at<Vec5f>(m, n)[k];
                    }
                }
            }
            res_layer.at<Vec5f>(i, j) = resVal;
        }
    }

    return res_layer;
}

cv::Mat Softmax(cv::Mat img) {
    int src_height = img.rows;
    int src_width = img.cols;
    cv::Mat res_layer = cv::Mat(cv::Size(src_width, src_height), CV_32FC(5));
    Vec5f max = Vec5f(0, 0, 0, 0, 0);

    for (int i = 0; i < src_height; i++) {
        for (int j = 0; j < src_width; j++) {
            for (int k = 0; k < 5; k++) {
                if (img.at<Vec5f>(i, j)[k] > max[k])
                    max[k] = img.at<Vec5f>(i, j)[k];
            }
        }
    }

    for (int i = 0; i < src_height; i++) {
        for (int j = 0; j < src_width; j++) {
            Vec5f t(0, 0, 0, 0, 0);
            float sum = 0;
            for (int k = 0; k < 5; k++) {
                t[k] = exp(img.at<Vec5f>(i, j)[k] - max[k]);
                sum += t[k];
            }
            for (int k = 0; k < 5; k++) {
                res_layer.at<Vec5f>(i, j)[k] = t[k] / sum;
            }
        }
    }

    return res_layer;
}