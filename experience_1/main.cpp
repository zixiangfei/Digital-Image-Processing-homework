#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;


string home="/Users/zhengluzhou/Desktop/study/Digital Image Processing/experience/experience_1/"; //工作目录

void v_contrast_change(int contrastValue,void *img) {
    Mat src = *((Mat *)img);
    int height = src.rows, width = src.cols;
    Mat hsv = Mat::zeros(src.size(),CV_8UC3);
    cvtColor(src,hsv,COLOR_BGR2HSV); //转化为hsv图像
    //imshow("hsv",hsv);
    vector<Mat> mv;
    split(hsv,mv);//分离出来三个通道
    double contrastRadio = contrastValue / 100.0;
    for(int row=0;row<height;++row) {
        for(int col=0;col<width;++col) {
            mv[2].at<uchar>(row,col) = pow(mv[2].at<uchar>(row,col)/255.0,contrastRadio)*255.0; //对亮度通道每个元素进行伽马变换
        }
    }
    merge(mv,hsv); //合并通道
    Mat dst;
    cvtColor(hsv,dst,COLOR_HSV2BGR); //得到处理好的图像
    imshow("v_contrast_change",dst);
}

void gray_contrast_change(int contrastValue,void *img) {
    Mat src = *((Mat *)img);
    int height = src.rows, width = src.cols;
    Mat gray;
    cvtColor(src,gray,COLOR_BGR2GRAY);
    double contrastRadio = contrastValue / 100.0;
    for(int row=0;row<height;++row) {
        for(int col=0;col<width;++col) {
            gray.at<uchar>(row,col) = pow(gray.at<uchar>(row,col)/255.0,contrastRadio) *255.0;
        }
    }
    imshow("gray_contrast_change",gray);
}

void homework1() {
    Mat src = imread(home+"experience1.JPG"); //获得原始图片
    namedWindow("v_contrast_change",WINDOW_AUTOSIZE);
    namedWindow("gray_contrast_change",WINDOW_AUTOSIZE);
    int v_contrastValue = 100,gray_contrastValue = 100;
    createTrackbar("contrastValue","v_contrast_change",&v_contrastValue,500,v_contrast_change,(&src));
    createTrackbar("contrastValue","gray_contrast_change",&gray_contrastValue,500,gray_contrast_change,(&src));
    v_contrast_change(v_contrastValue,(&src));//初始化回调函数
    gray_contrast_change(v_contrastValue,(&src));
    imshow("src",src);
    waitKey(0);
}

void ksize_change(int ksize,void *img) {
    Mat src = *((Mat *)img);
    Mat dst;
    if(ksize!=0) {
        blur(src,dst,Size(ksize,ksize),Point(-1,-1));
        imshow("blur",dst);
    }
    else {
        imshow("blur",src);
    }
}

void homework2() {
    Mat src = imread(home+"experience1_2.JPG");
    namedWindow("src",WINDOW_AUTOSIZE);
    imshow("src",src);
    namedWindow("blur",WINDOW_AUTOSIZE);
    int ksize = 1;
    createTrackbar("ksize","blur",&ksize,20,ksize_change,(&src));
    ksize_change(ksize,(&src));
    waitKey(0);
}

void homework3() {
    Mat src = imread(home+"experience1_3.JPG");
    Mat kernel1_Gx = (Mat_<char>(3,3)<<-1,0,1,-2,0,2,-1,0,1);
    Mat kernel1_Gy = (Mat_<char>(3,3)<<1,2,1,0,0,0,-1,-2,-1);
    Mat kernel2_Gx = (Mat_<char>(3,3)<<-1,0,1,-1,0,1,-1,0,1);
    Mat kernel2_Gy = (Mat_<char>(3,3)<<1,1,1,0,0,0,-1,-1,-1);
    //Mat kernel3 = (Mat_<char>(3,3)<<-1,-1,2,-1,2,-1,2,-1,-1);
    Mat kernel3_1 = (Mat_<char>(3,3)<<2,-1,-1,-1,2,-1,-1,-1,2);
    Mat kernel3_2 = (Mat_<char>(3,3)<<-1,-1,2,-1,2,-1,2,-1,-1);
    Mat dst1,dst2,dst3,dst3_1,dst3_2,dst1_x,dst1_y,dst2_x,dst2_y;
    namedWindow("src",WINDOW_AUTOSIZE);
    imshow("src",src);
    filter2D(src,dst1_x,-1,kernel1_Gx);
    filter2D(src,dst1_y,-1,kernel1_Gy);
    addWeighted(dst1_x,0.5,dst1_y,0.5,0,dst1);
    filter2D(src,dst2_x,-1,kernel2_Gx);
    filter2D(src,dst2_y,-1,kernel2_Gy);
    addWeighted(dst2_x,0.5,dst2_y,0.5,0,dst2);
    filter2D(src,dst3_1,-1,kernel3_1);
    //filter2D(src,dst3_2,-1,kernel3_2);
    //addWeighted(dst3_1,0.5,dst3_2,0.5,0,dst3);
    namedWindow("kernel1",WINDOW_AUTOSIZE);
    imshow("kernel1",dst1);
    namedWindow("kernel2",WINDOW_AUTOSIZE);
    imshow("kernel2",dst2);
    namedWindow("kernel3",WINDOW_AUTOSIZE);
    imshow("kernel3",dst3_1);
    waitKey(0);
}

int main() {
    //homework1();
    //homework2();
    homework3();
    return 0;
}