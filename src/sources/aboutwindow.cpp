#include "aboutwindow.h"

AboutWindow::AboutWindow(QWidget *parent) : QDialog(parent)
{
    di = new Ui::Dialog;
    di->setupUi(this);
}

AboutWindow::~AboutWindow()
{
}