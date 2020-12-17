//
// Created by 22343 on 2020/12/17.
//

#ifndef IMAGEENHANCEMENTANDROID_SECEDCT_H
#define IMAGEENHANCEMENTANDROID_SECEDCT_H

#include <jni.h>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
#include <iostream>
#include <map>
#include <unordered_map>
#include <stdio.h>
#include <string.h>

using namespace std;
using namespace cv;

struct fkCDF {
    map<int, double> FkNorm;
    cv::Mat V;
};

int calTime(void(*contrastEnhancement)(cv::Mat, cv::Mat&), cv::Mat rawImg, cv::Mat& outImg){
    double startTime = static_cast<double>(cv::getTickCount());
    contrastEnhancement(rawImg, outImg);
    double endTime = static_cast<double>(cv::getTickCount());
    int seceTime = (endTime - startTime) * 1000 / cv::getTickFrequency();
    return seceTime;
}

cv::Mat readImg(string dir) {
    /*
    * This function will read image from the directory
    *
    * @param[in] dir the directory
    * @return the image, other is error
    */
    cv::Mat img = imread(dir);

    /* check for failure */
    if (img.empty()) {
        printf("Could not open or find the image\n");
        /* wait for any key press */
        exit(0);
    }
    return img;
}


int kNumOccurrences(cv::Mat& img, int k) {
    /*
    * This function will get the number of occurrences of the gray-level
    * k in the spatial image region.
    *
    * @param[img] the image
    * @param[k] the gray-level k
    * @return
    */
    int num = 0;

    int rows = img.rows; // rows of the matrix
    int cols = img.cols * img.channels(); //cols of the matrix
    for (int i = 0; i < rows; i++) {
        uchar* data = img.ptr<uchar>(i); // get the i th row address
        for (int j = 0; j < cols; j++) {
            int val = data[j];
            if (val == k) {
                num++;
            }
        }
    }
    return num;
}

cv::Mat hsvChannels(cv::Mat img) {
    /*
    * This function will get the three channels of HSV
    * @return
    */
    cv::Mat hsv;
    cv::cvtColor(img, hsv, COLOR_BGR2HSV);
    return hsv;

}


fkCDF spatialHistgram(cv::Mat hsv) {
    /*
    * This function will return 2D spatial histogram
    *
    * @param[in] img_hsv: the HSV image
    * @return the 2D spatial histogram
    */
    int H = hsv.rows;
    int W = hsv.cols;
    float ratio = H / float(W);
    int k = 256;

    long int total = H * W;
    //cout << "H: " << H << "; " << "W: " << W << ", " << H*W << endl;

    int M = round(pow(k * ratio, 0.5));
    int N = round(pow(k / ratio, 0.5));
    int rows = M;
    int cols = N;

    //vector<Mat> channels;
    //split(hsv, channels);
    //cv::Mat raw = channels.at(2);
    cv::Mat raw;
    raw.create(H, W, CV_8UC1);

    if (hsv.isContinuous() && raw.isContinuous()) {
        H = 1;
        W = W * hsv.rows;
    }

    for (int i = 0; i < H; i++) {
        uchar* v = raw.ptr<uchar>(i);
        cv::Vec3b* data = hsv.ptr<cv::Vec3b>(i);
        #pragma omp parallel for num_threads(4)
        for (int j = 0; j < W; j++) {
            v[j] = data[j][2];   // 17ms (4k)
        }
    }

    //for (int i = 0; i < 256; i++) {
    //	int nums = kNumOccurrences(raw, i);
    //	cout << "gray level: " << i << ", " << nums/float(total) << endl;
    //}
    //exit(0);

    cv::Mat src;
    resize(raw, src, Size(hsv.cols / 2, hsv.rows / 2), 0, 0, INTER_LINEAR);

    /* split the image into m*n blocks */
    vector<Mat> imgParts;
    int irows = src.rows, icols = src.cols; /* the rows and columns of the image*/
    int dr = irows / rows, dc = icols / cols; /* the rows and columns of the split image*/

    int delty = (irows % rows) / 2, deltx = (icols % cols) / 2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int x = j * dc + deltx, y = i * dr + delty;
            imgParts.push_back(src(cv::Rect(x, y, dc, dr)));
        }
    }

    /* the value k occurrences  */
    const int length = imgParts.size();
    unordered_map<int, Mat> histogram;
    unsigned int index = -1;
    int kk = 0;
    //#pragma omp parallel for num_threads(3)
    for (int i = 0; i < length; i++) {

        int rowRegion = imgParts[i].rows;
        int colRegion = imgParts[i].cols * imgParts[i].channels();
        ++index;
        if (imgParts[i].isContinuous()) {
            rowRegion = 1;
            colRegion = colRegion * imgParts[i].rows;
        }

        int histSize = 256;
        float range[] = { 0,256 };
        const float* histRange = { range };
        bool uniform = true, accumulate = false;
        Mat hist;
        calcHist(&imgParts[i], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

        histogram[i] = hist;
    }

    unordered_map<int, vector<int>> histBlocks;
    for (auto& it : histogram) {
        int indexBlock = it.first;
        Mat block = it.second;
        for (int i = 0; i < 256; i++) {
            histBlocks[indexBlock].push_back(block.at<float>(i));
        }
    }
    histogram.clear();

    unordered_map<int, vector<int>> hist;
    for (int i = 0; i < 256; i++) {
        int index = 0;
        for (auto& it : histBlocks) {
            hist[i].push_back(it.second[i]);
        }
    }
    histBlocks.clear();

    double eps = 0.00001;
    map<int, double> entropy;
    unordered_map<int, vector<int>>::iterator it;
    for (it = hist.begin(); it != hist.end(); it++) {
        int key = it->first;
        vector<int> vals = it->second;
        double sum = accumulate(vals.begin(), vals.end(), 0);
        double sk = 0.0;
        for (int ele : vals) {
            double val = ele;
            val = val / (sum + eps); /* normalize */
            if (val != 0) {
                sk += -(val * (log(val) / log(2)));
            }
        }
        entropy.insert(make_pair(key, sk));
    }

    double entropySum = 0.0;
    for (auto& it : entropy) { /* calculate the sum entropy */
        entropySum += it.second;
    }

    map<int, double> fk;
    for (auto& it : entropy) {
        double value = it.second / (entropySum - it.second + eps); /* compute a discrete function fk */
        fk.insert(make_pair(it.first, value));
    }
    double fkSum = 0.0;
    for (auto& it : fk) {
        fkSum += it.second;
    }

    map<int, double> fkNorm;
    for (auto& it : fk) {
        double value = it.second / (fkSum + eps);
        fkNorm.insert(make_pair(it.first, value));
    }

    int zeroNum = 0;
    for (auto& it : fkNorm) {
        int key = it.first;
        double val = it.second;
        if (val == 0) {
            ++zeroNum;
        } //cout << "key: " << key << "; " << "val: " << val << endl;
    }

    if (zeroNum >= 120 ) {
        fkCDF res;
        res.FkNorm = fkNorm;
        res.V = raw;
        return res;
    }

    map<int, double> cdf;
    double val = 0.0000;
    for (auto& it : fkNorm) {
        int key = it.first;
        val = val + it.second;
        cdf.insert(make_pair(key, val));
    }

    /* mapping function: using the cumulative distribution function */
    int yu = 255;
    int yd = 0;
    map<int, int> ymap;
    for (auto& it : cdf) {
        int value = round(it.second * (yu - yd) + yd);
        ymap.insert(make_pair(it.first, value));
    }

    /* get the globally enhanced image */
    cv::Mat image;
    image.create(raw.rows, raw.cols, CV_8UC1);

    uchar lutData[256];
    int i = 0;
    for (auto& it : ymap) {
        lutData[i] = it.second;
        i++;
    }

    Mat lut(1, 256, CV_8UC1, lutData);
    cv::LUT(raw, lut, image);

    fkCDF res;
    res.FkNorm = fkNorm;
    res.V = image;
    return res;
}

cv::Mat dctTransform(cv::Mat globalImg) {
    /*
    * This function will transform image into the directory
    *
    * @param[in] dir the directory
    * @return
    */
    globalImg.convertTo(globalImg, CV_64F);

    cv::Mat imgDCT;
    cv::dct(globalImg, imgDCT);

    return imgDCT;
}

cv::Mat domainCoefweight(cv::Mat imgDct, map<int, double> fkNorm) {
    /*
    * This function will transform domain coefficient weighting
    *
    * @return
    */
    int H = imgDct.rows;
    int W = imgDct.cols;
    double sum = 0.0;
    for (auto& it : fkNorm) {
        if (it.second != 0) {
            sum += -it.second * (log(it.second) / log(2));
        }
    }

    cv::Mat imgWeight;
    imgWeight.create(H, W, CV_64F);
    double alpha = pow(sum, 0.20);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            double weight = (1.0 + (alpha - 1.0) * i / float(H - 1.0)) *
                            (1.0 + (alpha - 1.0) * j / float(W - 1.0));
            imgWeight.at<double>(i, j) = weight * imgDct.at<double>(i, j);
        }
    }

    return imgWeight;
}

cv::Mat inverseDct(cv::Mat imgWeight) {
    /*
    * This function will perform the inverse 2D-DCT transform
    * @return
    */
    imgWeight.convertTo(imgWeight, CV_64FC1);

    cv::Mat imgIDCT;
    cv::idct(imgWeight, imgIDCT);
    imgIDCT.convertTo(imgIDCT, CV_8U);

    return imgIDCT;
}

cv::Mat colorRestoration(cv::Mat& hsv, cv::Mat V) {
    /*
    * This function will perform the color restoration
    * @return
    */

    int rows = V.rows;
    int cols = V.cols;

    if (hsv.isContinuous() && V.isContinuous()) {
        rows = 1;
        cols = cols * V.rows;
    }

    for (int i = 0; i < rows; i++) {
        uchar* data = V.ptr<uchar>(i);
        cv::Vec3b* data3b = hsv.ptr<cv::Vec3b>(i);
        #pragma omp parallel for num_threads(4)
        for (int j = 0; j < cols; j++) {
            data3b[j][2] = data[j];
        }
    }

    cv::Mat newImg;
    cv::cvtColor(hsv, newImg, COLOR_HSV2BGR); /* convert the hsv to rgb */

    return newImg;
}

void contrastEnhancement(cv::Mat input, cv::Mat& output) {
    /*
    * This function will enhance contrast of the input image
    * @return: the enhanced image
    */
    /* image contrast enhancement */

    cv::Mat hsv = hsvChannels(input);
    fkCDF res = spatialHistgram(hsv);

    //cv::Mat imgDCT = dctTransform(res.V);
    //cv::Mat imgWeight = domainCoefweight(imgDCT, res.FkNorm);
    //cv::Mat imgIDCT = inverseDct(imgWeight);
    output = colorRestoration(hsv, res.V);

}

#endif //IMAGEENHANCEMENTANDROID_SECEDCT_H




