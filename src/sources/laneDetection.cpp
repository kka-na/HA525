#include "laneDetection.h"

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>

#include <opencv2/imgproc.hpp>

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

    cap >> frame;

    int width = frame.cols;
    int height = frame.rows;

    Mat birdEyeView, scharrFiltered, colorFiltered, filtered, laneSearched, laneMask, final;
    Mat temp, tempImg, perspMatrix;

    Point p1s = Point2f(width / 2 - upX_diff, height / 2 - upY_diff);
    Point p2s = Point2f(width / 2 + upX_diff, height / 2 - upY_diff);
    Point p3s = Point2f(width / 2 - downX_diff, height / 2 + downY_diff);
    Point p4s = Point2f(width / 2 + downX_diff, height / 2 + downY_diff);

    Point p1d = Point2f(dstX, 0);
    Point p2d = Point2f(width - dstX, 0);
    Point p3d = Point2f(dstX, height);
    Point p4d = Point2f(width - dstX, height);

    vector<Point2f> srcRectCoord = {p1s, p2s, p3s, p4s};
    vector<Point2f> dstRectCoord = {p1d, p2d, p3d, p4d};

    std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();

    int frame_cnt = 0;
    while (running)
    {
        cap >> frame; // 0

        if (frame.empty())
            break;

        emitProcesses(frame, s, 0);

        // some processing
        birdEyeView = BirdEyeViewTransform(frame); // 1
        emitProcesses(birdEyeView, s, 1);
        scharrFiltered = ScharrFilter(birdEyeView); // 2
        emitProcesses(scharrFiltered, s, 2);
        colorFiltered = ColorFilter(birdEyeView); // 3
        emitProcesses(colorFiltered, s, 3);
        filtered = addFrame(scharrFiltered, colorFiltered); // 4
        emitProcesses(filtered, s, 4);
        laneSearched = SlidingWindow(filtered); // 5
        // imshow("laneSearched", laneSearched);
        emitProcesses(laneSearched, s, 5);
        laneMask = InvBirdEyeViewTransform(laneSearched); // 6
        // imshow("laneMask", laneMask);
        emitProcesses(laneMask, s, 6);
        final = maskingLane(frame, laneMask); // 7
        // imshow("final", final);
        emitProcesses(final, s, 7);
        // waitKey(sleep_cnt);

        s = std::chrono::high_resolution_clock::now();

        frame_cnt += 1;
        if (frame_cnt == sleep_cnt)
        {
            emit updateSlider();
            frame_cnt = 0;
        }
        QThread::msleep(sleep_cnt);
    }
}

QImage LaneDetection::convert2QT(Mat _frame)
{
    Mat frame;
    cvtColor(_frame, frame, cv::COLOR_BGR2RGB);
    QImage qimage(frame.cols, frame.rows, QImage::Format_RGB888);
    memcpy(qimage.scanLine(0), frame.data, static_cast<size_t>(qimage.width() * qimage.height() * frame.channels()));
    return qimage;
}

void LaneDetection::emitProcesses(Mat frame, std::chrono::high_resolution_clock::time_point s, int type)
{
    QImage qimage = convert2QT(frame);
    auto f = std::chrono::high_resolution_clock::now();
    int fps = 1000 / (float(std::chrono::duration_cast<std::chrono::milliseconds>(f - s).count()));

    emit sendImage(qimage, type);
    emit sendFPS(fps, type);
}

// Perspective Transform using OpenCV
Mat LaneDetection::BirdEyeViewTransform(Mat src)
{
    Mat perspMatrix, dst;
    int width = src.cols;
    int height = src.rows;
    vector<Point2f> srcRectCoord, dstRectCoord;

    // Source Point
    Point p1s = Point2f(width / 2 - upX_diff, height / 2 - upY_diff);
    Point p2s = Point2f(width / 2 + upX_diff, height / 2 - upY_diff);
    Point p3s = Point2f(width / 2 - downX_diff, height / 2 + downY_diff);
    Point p4s = Point2f(width / 2 + downX_diff, height / 2 + downY_diff);

    // Destination Point
    Point p1d = Point2f(dstX, 0);
    Point p2d = Point2f(width - dstX, 0);
    Point p3d = Point2f(dstX, height);
    Point p4d = Point2f(width - dstX, height);

    srcRectCoord = {p1s, p2s, p3s, p4s};
    dstRectCoord = {p1d, p2d, p3d, p4d};

    // get 3X3 Homographic Matrix
    perspMatrix = getPerspectiveTransform(srcRectCoord, dstRectCoord);

    // Perspective Transform
    warpPerspective(src, dst, perspMatrix, Size(width, height), INTER_LINEAR);

    return dst;
}

// Perspective Transform
Mat LaneDetection::myPerspectiveTransform(Mat src, vector<Point2f> srcRectCoord, vector<Point2f> dstRectCoord)
{
    Mat dst = Mat::zeros(src.size(), src.type());
    int width = src.cols;
    int height = src.rows;
    float new_w, new_h;

    // get Homographic Matrix
    Mat perspMatrix = getPerspectiveTransform(srcRectCoord, dstRectCoord);

    // get Inverse Homographic Matrix
    Mat invPerspMatrix = perspMatrix.inv();

    float h_Matrix[] = {perspMatrix.at<float>(0), perspMatrix.at<float>(1), perspMatrix.at<float>(2),
                        perspMatrix.at<float>(3), perspMatrix.at<float>(4), perspMatrix.at<float>(5),
                        perspMatrix.at<float>(6), perspMatrix.at<float>(7), perspMatrix.at<float>(8)};

    float h_Matrix_inv[] = {invPerspMatrix.at<float>(0), invPerspMatrix.at<float>(1), invPerspMatrix.at<float>(2),
                            invPerspMatrix.at<float>(3), invPerspMatrix.at<float>(4), invPerspMatrix.at<float>(5),
                            invPerspMatrix.at<float>(6), invPerspMatrix.at<float>(7), invPerspMatrix.at<float>(8)};

    // Perspective Transform
    for (int h = 0; h < height; h++)
    {
        Vec3b *srcPtr = src.ptr<Vec3b>(h);
        for (int w = 0; w < width; w++)
        {
            new_w = (h_Matrix[0] * w + h_Matrix[1] * h + h_Matrix[2]) / (h_Matrix[6] * w + h_Matrix[7] * h + h_Matrix[8]);
            new_h = (h_Matrix[3] * w + h_Matrix[4] * h + h_Matrix[5]) / (h_Matrix[6] * w + h_Matrix[7] * h + h_Matrix[8]);
            if (0 <= new_h && new_h < height && 0 <= new_w && new_w < width)
            {
                Vec3b *dstPtr = dst.ptr<Vec3b>(new_h);
                for (int k = 0; k < 3; k++)
                {
                    dstPtr[(int)new_w][k] = srcPtr[w][k];
                }
            }
        }
    }

    float a, b;
    int i, j;

    // Interpolation
    for (int h = 0; h < height; h++)
    {
        Vec3b *dstPtr = dst.ptr<Vec3b>(h);
        for (int w = 0; w < width; w++)
        {
            if (dstPtr[w] == Vec3b(0, 0, 0))
            {
                new_w = (h_Matrix_inv[0] * w + h_Matrix_inv[1] * h + h_Matrix_inv[2]) / (h_Matrix_inv[6] * w + h_Matrix_inv[7] * h + h_Matrix_inv[8]);
                new_h = (h_Matrix_inv[3] * w + h_Matrix_inv[4] * h + h_Matrix_inv[5]) / (h_Matrix_inv[6] * w + h_Matrix_inv[7] * h + h_Matrix_inv[8]);
                if (0 <= new_h && new_h < height && 0 <= new_w && new_w < width)
                {
                    Vec3b *srcPtr1 = src.ptr<Vec3b>(new_h);
                    Vec3b *srcPtr2 = src.ptr<Vec3b>(new_h + 1);
                    i = new_w;
                    j = new_h;
                    a = new_w - i;
                    b = new_h - j;
                    for (int k = 0; k < 3; k++)
                    {
                        dstPtr[w][k] = (1 - a) * (1 - b) * srcPtr1[i][k] + a * (1 - b) * srcPtr1[i + 1][k] + a * b * srcPtr2[i + 1][k] + (1 - a) * b * srcPtr2[i][k];
                    }
                }
            }
        }
    }

    return dst;
}

// Scharr Filtering for Edge Detection
Mat LaneDetection::ScharrFilter(Mat src)
{
    int width = src.cols;
    int height = src.rows;
    Mat src_gray;

    cvtColor(src, src_gray, COLOR_BGR2GRAY);

    Mat zeroPad = Mat::zeros(height + 2, width + 2, src_gray.type());
    Mat zeroPadDst = Mat::zeros(height, width, src_gray.type());

    // Zeropadding
    for (int h = 0; h < height; h++)
    {
        uchar *zPadPtr = zeroPad.ptr<uchar>(h + 1);
        uchar *srcPtr = src_gray.ptr<uchar>(h);
        for (int w = 0; w < width; w++)
        {
            zPadPtr[w + 1] = srcPtr[w];
        }
    }

    float scharrFilter_x[] = {3, 10, 3, 0, 0, 0, -3, -10, -3};
    float scharrFilter_y[] = {3, 0, -3, 10, 0, -10, 3, 0, -3};

    // Convolution
    for (int h = 1; h < height + 1; h++)
    {
        uchar *zPadPtr_1 = zeroPad.ptr<uchar>(h - 1);
        uchar *zPadPtr_2 = zeroPad.ptr<uchar>(h);
        uchar *zPadPtr_3 = zeroPad.ptr<uchar>(h + 1);
        uchar *zPadDstPrt = zeroPadDst.ptr<uchar>(h - 1);
        for (int w = 1; w < width + 1; w++)
        {
            int sum = 0;
            // int idx = 0;
            sum = zPadPtr_1[w - 1] * scharrFilter_x[0] + zPadPtr_1[w] * scharrFilter_x[1] + zPadPtr_1[w + 1] * scharrFilter_x[2] + zPadPtr_2[w - 1] * scharrFilter_x[3] + zPadPtr_2[w] * scharrFilter_x[4] + zPadPtr_2[w + 1] * scharrFilter_x[5] + zPadPtr_3[w - 1] * scharrFilter_x[6] + zPadPtr_3[w] * scharrFilter_x[7] + zPadPtr_3[w + 1] * scharrFilter_x[8] + zPadPtr_1[w - 1] * scharrFilter_y[0] + zPadPtr_1[w] * scharrFilter_y[1] + zPadPtr_1[w + 1] * scharrFilter_y[2] + zPadPtr_2[w - 1] * scharrFilter_y[3] + zPadPtr_2[w] * scharrFilter_y[4] + zPadPtr_2[w + 1] * scharrFilter_y[5] + zPadPtr_3[w - 1] * scharrFilter_y[6] + zPadPtr_3[w] * scharrFilter_y[7] + zPadPtr_3[w + 1] * scharrFilter_y[8];
            if (sum < 0)
                sum = (-1) * sum;
            if (sum > 255)
                sum = 255;
            if (sum < 150)
                sum = 0;
            else
                sum = 255;
            zPadDstPrt[w - 1] = (uchar)sum;
        }
    }

    return zeroPadDst;
}

// Color Filtering
Mat LaneDetection::ColorFilter(Mat src)
{
    int width = src.cols;
    int height = src.rows;
    Mat src_hsv;
    Mat dst = Mat::zeros(height, width, 0);
    cvtColor(src, src_hsv, COLOR_BGR2HSV);

    // Pixel by Pixel Color Filtering
    for (int h = 0; h < height; h++)
    {
        Vec3b *ptr = src_hsv.ptr<Vec3b>(h);
        uchar *dstPtr = dst.ptr<uchar>(h);

        for (int w = 0; w < width; w++)
        {

            // Yellow Color Filtering
            if (lower_yellow[0] <= ptr[w][0] && ptr[w][0] <= upper_yellow[0] && lower_yellow[1] <= ptr[w][1] && ptr[w][1] <= upper_yellow[1] && lower_yellow[2] <= ptr[w][2] && ptr[w][2] <= upper_yellow[2])
            {
                dstPtr[w] = 255;
            }
            // White Color Filtering
            else if (lower_white[0] <= ptr[w][0] && ptr[w][0] <= upper_white[0] && lower_white[1] <= ptr[w][1] && ptr[w][1] <= upper_white[1] && lower_white[2] <= ptr[w][2] && ptr[w][2] <= upper_white[2])
            {
                dstPtr[w] = 255;
            }
            else
            {
                dstPtr[w] = 0;
            }
        }
    }

    return dst;
}

// add Color Filtering Result and Scharr Filtering Result
Mat LaneDetection::addFrame(Mat src1, Mat src2)
{
    int width = src1.cols;
    int height = src1.rows;
    Mat dst = Mat::zeros(height, width, 0);

    // add task Pixel by Pixel
    for (int h = 0; h < height; h++)
    {
        uchar *src1Ptr = src1.ptr<uchar>(h);
        uchar *src2Ptr = src2.ptr<uchar>(h);
        uchar *dstPtr = dst.ptr<uchar>(h);
        for (int w = 0; w < width; w++)
        {
            dstPtr[w] = src1Ptr[w] + src2Ptr[w];
        }
    }

    return dst;
}

// Sliding Window for Lane Searching
Mat LaneDetection::SlidingWindow(Mat src)
{
    Mat dst;

    int width = src.cols;
    int height = src.rows;

    Mat laneMask = Mat::zeros(Size(width, height), CV_8UC3);

    // number of grid
    int gridX_number = 20;
    int gridY_number = 10;

    // searching grid for each lane(left & right)
    int left_grid = 3;
    int right_grid = 3;

    Size grid_size(width / gridX_number, height / gridY_number);

    cvtColor(src, dst, COLOR_GRAY2BGR);

    // Initialize Searching Point on First Frame
    if (searchingPoint[0] == 0 && searchingPoint[1] == 0)
    {
        Point startIndex = FindStartIndex(src);
        searchingPoint[0] = (startIndex.x / grid_size.width) * grid_size.width;
        searchingPoint[1] = (startIndex.y / grid_size.width) * grid_size.width;
    }

    // Setting first Searching point in each frame
    Point curLeftGrid = Point2i(searchingPoint[0], height - grid_size.height);
    Point curRightGrid = Point2i(searchingPoint[1], height - grid_size.height);
    Point nextLeftGrid, nextRightGrid;
    Point targetGrid;

    int left_score;
    int right_score;
    int score_temp;

    bool leftFind = true;
    bool rightFind = true;

    // Start Searching from the bottom of the frame to the top of the frame
    for (int i = 0; i < gridY_number; i++)
    {
        left_score = 0;
        right_score = 0;

        // Searching Point of Next Frame
        if (i == 2)
        {
            searchingPoint[0] = curLeftGrid.x;
            searchingPoint[1] = curRightGrid.x;
        }

        // if there is no lane, 5 grid will search the lane
        if (leftFind == false)
            left_grid = 5;
        else
            left_grid = 3;

        if (rightFind == false)
            right_grid = 5;
        else
            right_grid = 3;

        // left lane searching
        for (int left = 0; left < left_grid; left++)
        {
            targetGrid = Point(curLeftGrid.x + (left - (left_grid / 2)) * grid_size.width, curLeftGrid.y);
            if (0 <= targetGrid.x && targetGrid.x + grid_size.width <= width && 0 <= targetGrid.y && targetGrid.y + grid_size.height <= height)
            {
                // Calculate each grid's score
                score_temp = calcGridScore(src, targetGrid, grid_size);
                if (left_score < score_temp)
                {
                    left_score = score_temp;
                    nextLeftGrid = targetGrid;
                }
                rectangle(laneMask, Rect(targetGrid.x, targetGrid.y, grid_size.width, grid_size.height), Scalar(0, 0, 255), 2);
            }
        }
        if (left_score > 50000)
            rectangle(laneMask, Rect(nextLeftGrid.x, targetGrid.y, grid_size.width, grid_size.height), Scalar(0, 0, 255), FILLED);
        curLeftGrid.x = nextLeftGrid.x;
        curLeftGrid.y = curLeftGrid.y - grid_size.height;

        // right lane searching
        for (int right = 0; right < right_grid; right++)
        {
            targetGrid = Point(curRightGrid.x + (right - (right_grid / 2)) * grid_size.width, curRightGrid.y);
            if (0 <= targetGrid.x && targetGrid.x + grid_size.width <= width && 0 <= targetGrid.y && targetGrid.y + grid_size.height <= height)
            {
                // Calculate each grid's score
                score_temp = calcGridScore(src, targetGrid, grid_size);
                if (right_score < score_temp)
                {
                    right_score = score_temp;
                    nextRightGrid = targetGrid;
                }
                rectangle(laneMask, Rect(targetGrid.x, targetGrid.y, grid_size.width, grid_size.height), Scalar(255, 0, 0), 2);
            }
        }
        if (right_score > 50000)
            rectangle(laneMask, Rect(nextRightGrid.x, targetGrid.y, grid_size.width, grid_size.height), Scalar(255, 0, 0), FILLED);
        curRightGrid.x = nextRightGrid.x;
        curRightGrid.y = curRightGrid.y - grid_size.height;

        // if there is no lane
        if (left_score < 50000)
            leftFind = false;
        else
            leftFind = true;
        if (right_score < 50000)
            rightFind = false;
        else
            rightFind = true;
    }

    return laneMask;
}

// Searching first Searching Point from Histogram of frame
Point LaneDetection::FindStartIndex(Mat src)
{
    int width = src.cols;
    int leftMaxIndex = 0;
    int rightMaxIndex = width / 2;
    float maxValue = 0;
    Mat histogram;

    // Calculate each column's Sum
    reduce(src, histogram, 0, REDUCE_SUM, CV_32S);

    // Left lane
    for (int i = 0; i < width / 2; i++)
    {
        if (maxValue < histogram.at<int>(i))
        {
            maxValue = histogram.at<int>(i);
            leftMaxIndex = i;
        }
    }

    maxValue = 0;
    // Right lane
    for (int i = width / 2; i < width; i++)
    {
        if (maxValue < histogram.at<int>(i))
        {
            maxValue = histogram.at<int>(i);
            rightMaxIndex = i;
        }
    }

    return Point2i(leftMaxIndex, rightMaxIndex);
}

// Calculate grid score
int LaneDetection::calcGridScore(Mat src, Point grid, Size grid_size)
{
    int score = 0;
    int width = grid_size.width;
    int height = grid_size.height;

    // Add every Pixel's value in each grid
    for (int h = 0; h < height; h++)
    {
        uchar *ptr = src.ptr<uchar>(grid.y + h);
        for (int w = 0; w < width; w++)
        {
            score += (int)ptr[grid.x + w];
        }
    }

    return score;
}

Mat LaneDetection::InvBirdEyeViewTransform(Mat src)
{
    Mat perspMatrix, dst;
    int width = src.cols;
    int height = src.rows;
    vector<Point2f> srcRectCoord, dstRectCoord;

    Point p1s = Point2f(dstX, 0);
    Point p2s = Point2f(width - dstX, 0);
    Point p3s = Point2f(dstX, height);
    Point p4s = Point2f(width - dstX, height);

    Point p1d = Point2f(width / 2 - upX_diff, height / 2 - upY_diff);
    Point p2d = Point2f(width / 2 + upX_diff, height / 2 - upY_diff);
    Point p3d = Point2f(width / 2 - downX_diff, height / 2 + downY_diff);
    Point p4d = Point2f(width / 2 + downX_diff, height / 2 + downY_diff);

    srcRectCoord = {p1s, p2s, p3s, p4s};
    dstRectCoord = {p1d, p2d, p3d, p4d};

    perspMatrix = getPerspectiveTransform(srcRectCoord, dstRectCoord);
    warpPerspective(src, dst, perspMatrix, Size(width, height), INTER_LINEAR);

    return dst;
}

// Add lane mask and source frame
Mat LaneDetection::maskingLane(Mat src, Mat lane)
{
    int width = src.cols;
    int height = src.rows;

    for (int h = 0; h < height; h++)
    {
        Vec3b *lanePtr = lane.ptr<Vec3b>(h);
        Vec3b *srcPtr = src.ptr<Vec3b>(h);
        for (int w = 0; w < width; w++)
        {
            if (lanePtr[w] != Vec3b(0, 0, 0))
            {
                srcPtr[w] = lanePtr[w];
            }
        }
    }
    return src;
}
LaneDetection::~LaneDetection()
{
}
