#ifndef LANEDETECTION_H
#define LANEDETECTION_H
#pragma once

#include <QThread>
#include <QObject>
#include <QImage>
#include <chrono>

#include "opencv2/opencv.hpp"

#include <string>

void gpu_ColorFilter(uchar *, uchar *, int, int, int);
void gpu_AddFrame(uchar *, uchar *, uchar *, int, int);
void gpu_MaskingLane(uchar *, uchar *, int, int, int);

class LaneDetection : public QThread
{
    Q_OBJECT

public:
    LaneDetection(QObject *parent = 0);
    ~LaneDetection();

public:
    void setStart();
    std::string data_path;
    bool with_cuda = false;
    bool running = true;
    bool connect = false;
    bool first_cnt = true;

private:
    cv::VideoCapture cap;
    int sleep_cnt = 33;

    int searchingPoint[2] = {0, 0};
    int upX_diff = 90;
    int upY_diff = 150;
    int downX_diff = 380;
    int downY_diff = 100;
    int dstX = 0;

    cv::Scalar lower_yellow = cv::Scalar(15, 40, 100);
    cv::Scalar upper_yellow = cv::Scalar(23, 255, 255);

    cv::Scalar lower_white = cv::Scalar(0, 0, 200);
    cv::Scalar upper_white = cv::Scalar(255, 35, 255);

private:
    void startDetection();
    QImage convert2QT(cv::Mat);
    void emitProcesses(cv::Mat, std::chrono::high_resolution_clock::time_point, int);

    cv::Mat BirdEyeViewTransform(cv::Mat);
    cv::Mat myPerspectiveTransform(cv::Mat, std::vector<cv::Point2f>, std::vector<cv::Point2f>);
    cv::Mat ScharrFilter(cv::Mat);
    cv::Mat ColorFilter(cv::Mat);
    cv::Mat addFrame(cv::Mat, cv::Mat);
    cv::Mat SlidingWindow(cv::Mat);
    cv::Point FindStartIndex(cv::Mat);
    int calcGridScore(cv::Mat, cv::Point, cv::Size);
    cv::Mat InvBirdEyeViewTransform(cv::Mat);
    cv::Mat maskingLane(cv::Mat, cv::Mat);

signals:
    void sendDuration(int);
    void sendImage(QImage, int);
    void sendFPS(int, int);
    void updateSlider();
};

#endif
