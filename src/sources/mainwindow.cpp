#include "mainwindow.h"

#include <iostream>
#include <memory>
#include <fstream>
#include <string>
#include <unistd.h>
#include <algorithm>

#include <QCoreApplication>
#include <QListWidgetItem>
#include <QMessageBox>
#include <QFileDialog>
#include <QInputDialog>
#include <QFile>
#include <QDir>
#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QMetaType>
#include <QStorageInfo>
#include <QtCharts>

using namespace std;

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);
	this->setFunction();
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::setFunction()
{
	ld = new LaneDetection(this);

	connect(ui->openButton, SIGNAL(clicked()), this, SLOT(setOpen()));
	connect(ui->connectButton, SIGNAL(clicked()), this, SLOT(setConnect()));
	connect(ui->startButton, SIGNAL(clicked()), this, SLOT(setStart()));
	connect(ui->stopButton, SIGNAL(clicked()), this, SLOT(setStop()));
	connect(ui->actionAbout_HA525, SIGNAL(triggered()), this, SLOT(setAbout()));

	connect(ld, SIGNAL(sendDuration(int)), this, SLOT(setSlider(int)));
	connect(ld, SIGNAL(sendImage(QImage, int)), this, SLOT(setImage(QImage, int)));
	connect(ld, SIGNAL(sendFPS(int, int)), this, SLOT(setFPS(int, int)));
	connect(ld, SIGNAL(updateSlider()), this, SLOT(updateSlider()));
}

void MainWindow::setAbout()
{
	AboutWindow *aw = new AboutWindow();
	aw->setModal(true);
	aw->show();
}

void MainWindow::setSlider(int duration)
{
	ui->horizontalSlider->setRange(0, duration);
	ui->horizontalSlider->setSingleStep(1);
	ui->labelTime->setText(QString::number(duration).append(" sec"));
	sec = 0;
}

void MainWindow::updateSlider()
{
	sec += 1;
	ui->horizontalSlider->setValue(sec);
}

void MainWindow::setDisplay()
{
	ui->openButton->setStyleSheet("QPushButton{color: #ffffff; background-color: #000000;}");
	ui->connectButton->setStyleSheet("QPushButton{color: #ffffff; background-color: #000000;}");
	ui->startButton->setStyleSheet("QPushButton{color: #ffffff; background-color: #000000;}");
}

void MainWindow::setOpen()
{
	ui->openButton->setStyleSheet("QPushButton{color: #000000; background-color: #5280ff;}");

	QString getPath = QFileDialog::getOpenFileName(this, "Select File of Lane Video", QDir::currentPath());
	data_path = getPath.toStdString();
	ld->data_path = data_path;
}

void MainWindow::setConnect()
{
	ui->connectButton->setStyleSheet("QPushButton{color: #000000; background-color: #5280ff;}");
	ld->connect = true;
}

void MainWindow::setStart()
{
	ui->startButton->setStyleSheet("QPushButton{color: #000000; background-color: #27b879;}");
	ui->stopButton->setStyleSheet("QPushButton{color: #ffffff; background-color: #000000;}");
	ld->setStart();
}

void MainWindow::setStop()
{
	ui->stopButton->setStyleSheet("QPushButton{color: #ffffff; background-color: #ad2828;}");
	ld->running = false;
	ld->connect = false;
	ld->first_cnt = true;
	pTimer->stop();
	ui->horizontalSlider->setValue(0);
	this->sec = 0;
	for (int i = 0; i < 8; i++)
	{
		QLabel *label = getLabel(i, 0);
		QLabel *labelFPS = getLabel(i, 1);
		label->clear();
		labelFPS->clear();
	}
	this->setDisplay();
}

void MainWindow::setImage(QImage _qimg, int type)
{
	QLabel *label = getLabel(type, 0);
	label->setPixmap(QPixmap::fromImage(_qimg).scaled(label->width(), label->height(), Qt::KeepAspectRatio));
	QCoreApplication::processEvents();
}

void MainWindow::setFPS(int _fps, int type)
{
	QLabel *label = getLabel(type, 1);
	label->setText(QString::number(_fps).append(" FPS"));
	QCoreApplication::processEvents();
}

void MainWindow::displayDetail()
{

	QWidget *dWindow = new QWidget;
	dWindow->resize(1280, 720);
	dWindow->setWindowTitle(QApplication::translate("displaywidget", "Display Widget"));
	dWindow->show();
	QLabel *dLabel = new QLabel(dWindow);
	dLabel->resize(1280, 720);
	dLabel->setPixmap(QPixmap::fromImage(detailImg).scaled(dLabel->width(), dLabel->height(), Qt::KeepAspectRatio));
	dLabel->show();
}

QLabel *MainWindow::getLabel(int _idx, int _type)
{
	QLabel *label;
	switch (_idx)
	{
	case 0:
		if (!_type)
			label = ui->label0;
		else
			label = ui->label0FPS;
		break;
	case 1:
		if (!_type)
			label = ui->label1;
		else
			label = ui->label1FPS;
		break;
	case 2:
		if (!_type)
			label = ui->label2;
		else
			label = ui->label2FPS;
		break;
	case 3:
		if (!_type)
			label = ui->label3;
		else
			label = ui->label3FPS;
		break;
	case 4:
		if (!_type)
			label = ui->label4;
		else
			label = ui->label4FPS;
		break;
	case 5:
		if (!_type)
			label = ui->label5;
		else
			label = ui->label5FPS;
		break;
	case 6:
		if (!_type)
			label = ui->label6;
		else
			label = ui->label6FPS;
		break;
	case 7:
		if (!_type)
			label = ui->label7;
		else
			label = ui->label7FPS;
		break;
	default:
		if (!_type)
			label = ui->label0;
		else
			label = ui->label0FPS;
		break;
	}
	return label;
}