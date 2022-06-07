#ifndef ABOUTWINDOW_H
#define ABOUTWINDOW_H

#include <QDialog>
#include "ui_aboutwindow.h"

class AboutWindow : public QDialog
{
public:
    AboutWindow(QWidget *parent = nullptr);
    ~AboutWindow();

private:
    Ui::Dialog *di;
};

#endif