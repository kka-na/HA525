#ifndef LANEDETECTION_H
#define LANEDETECTION_H
#pragma once

#include <QThread>
#include <QObject>
#include <QImage>

#include "opencv2/opencv.hpp"

#include <string>

class LaneDetection : public QThread
{
    Q_OBJECT

public:
    LaneDetection(QObject *parent = 0);
    ~LaneDetection();

public:
    void setStart();
    std::string data_path;
    bool running = true;
    bool connect = false;
    bool first_cnt = true;

private:
    cv::VideoCapture cap;
    int sleep_cnt = 33;
    void startDetection();
    QImage convert2QT(cv::Mat);

signals:
    void sendDuration(int);
    void sendImage(QImage, int);
    void sendFPS(int, int);
    void startSlider();
};

#endif
