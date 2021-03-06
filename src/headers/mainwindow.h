#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "ui_mainwindow.h"
#include "laneDetection.h"

#include <string>

#include <QMainWindow>
#include <QObject>
#include <QLabel>

#include "aboutwindow.h"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    class LaneDetection::LaneDetection *ld;

    QImage detailImg;
    int sec = 0;

    std::string data_path;

private:
    void setFunction();
    QLabel *getLabel(int, int);

private slots:
    void updateCUDA(bool);
    void setAbout();
    void setSlider(int);
    void updateSlider();
    void setDisplay();
    void setOpen();
    void setConnect();
    void setStart();
    void setStop();
    void setImage(QImage, int);
    void setFPS(int, int);
    void displayDetail();

private:
};
#endif // MAINWINDOW_H
