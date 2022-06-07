#include "laneDetection.h"

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>

#include <QFile>
#include <QDir>
#include <QTextStream>

using namespace cv;
using namespace std;

LaneDetection::LaneDetection(QObject *parent) : QThread(parent)
{
}

void LaneDetection::setStart()
{
    if (!connect)
        cap.open(data_path);
    else if (connect)
        cap.open(0);

    sleep_cnt = cap.get(CAP_PROP_FPS);
    float duration = float(cap.get(CAP_PROP_FRAME_COUNT)) / float(cap.get(CAP_PROP_FPS));
    emit sendDuration(int(duration));
    this->running = true;
    this->startDetection();
}

void LaneDetection::startDetection()
{
    Mat frame;
    while (running)
    {
        auto s1 = std::chrono::high_resolution_clock::now();

        cap >> frame;
        if (frame.empty())
            break;

        // some processing

        QImage qimage = convert2QT(frame);
        auto f1 = std::chrono::high_resolution_clock::now();
        int fps = 1000 / (float(std::chrono::duration_cast<std::chrono::milliseconds>(f1 - s1).count()));

        if (first_cnt)
        {
            emit startSlider();
            first_cnt = false;
        }
        for (int i = 0; i < 8; i++)
        {
            emit sendImage(qimage, i);
            emit sendFPS(fps, i);
        }

        QThread::msleep(sleep_cnt);
    }
}

QImage LaneDetection::convert2QT(Mat frame)
{
    cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    QImage qimage(frame.cols, frame.rows, QImage::Format_RGB888);
    memcpy(qimage.scanLine(0), frame.data, static_cast<size_t>(qimage.width() * qimage.height() * frame.channels()));
    return qimage;
}

LaneDetection::~LaneDetection()
{
}
