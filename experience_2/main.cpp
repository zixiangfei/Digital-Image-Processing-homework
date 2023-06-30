#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


string home="/Users/zhengluzhou/Desktop/study/Digital Image Processing/experience/experience_2/";

int fx[9]={0,1,1,-1,-1,0,0,1,-1};
int fy[9]={0,-1,1,-1,1,-1,1,0,0};

Mat filter(Mat src,Mat kernel_Gx,Mat kernel_Gy) {
    Mat dst = Mat::zeros(src.size(),src.type());
    int t=0.4*255;
    int row = dst.rows, col = dst.cols;
    for(int x=1;x<row-1;++x) {
        for(int y=1;y<col-1;++y) {
            int gray_Gx=0,gray_Gy=0;
            for(int k=0;k<9;++k) {
                int tx=fx[k]+x,ty=fy[k]+y;
                int kernel_x = 1 + fx[k], kernel_y = 1 + fy[k];
                gray_Gx += src.at<uchar>(tx,ty)*kernel_Gx.at<char>(kernel_x,kernel_y);
                gray_Gy += src.at<uchar>(tx,ty)*kernel_Gy.at<char>(kernel_x,kernel_y);
            }
            if(abs(gray_Gy)+abs(gray_Gx)<=t) {
                dst.at<uchar>(x,y)=0;
            }
            else {
                dst.at<uchar>(x,y)=255;
            }
        }
    }
    return dst;
}

void showManyPics(vector<Mat>pics,String windowName) {
    namedWindow(windowName,WINDOW_AUTOSIZE);
    int n=pics.size();
    int h=pics[0].rows,w=pics[0].cols;
    Mat dst = Mat::zeros(Size(w*n,h),pics[0].type());
    //cout<<dst.rows<<" "<<dst.cols<<'\n';
    for(int i=0;i<n;++i) {
        //cout<<0<<" "<<(i)*w<<" "<<w<<" "<<h<<'\n';
        pics[i].copyTo(dst(Rect(i*w,0,w,h)));
    }
    imshow(windowName,dst);
}

int main() {

    Mat src = imread(home+"experience2_2.JPG");//读取图像
    Mat img;
    cvtColor(src,img,COLOR_BGR2GRAY);
    Mat Sobel,Prewitt;
    Mat Sobel_Gx = (Mat_<char>(3,3)<<-1,-2,-1,0,0,0,1,2,1);
    Mat Sobel_Gy = (Mat_<char>(3,3)<<-1,0,1,-2,0,2,-1,0,1);
    Mat Prewitt_Gx = (Mat_<char>(3,3)<<-1,-1,-1,0,0,0,1,1,1);
    Mat Prewitt_Gy = (Mat_<char>(3,3)<<-1,0,1,-1,0,1,-1,0,1);
    Prewitt = filter(img,Prewitt_Gx,Prewitt_Gy);
    Sobel = filter(img,Sobel_Gx,Sobel_Gy);
    vector<Mat> pics;
    pics.push_back(img);
    pics.push_back(Sobel);
    pics.push_back(Prewitt);
    showManyPics(pics,"src Soble Prewitt");

    Mat Hough;
    src.copyTo(Hough);
    vector<Vec4f> lines;
    //HoughLines(Prewitt,lines,1,CV_PI/180.0,150,0,0);
    HoughLinesP(Sobel,lines,1,CV_PI/180,250,0,0);
    for(auto line : lines) {
        int x1=line[0],y1=line[1],x2=line[2],y2=line[3];
        cv::line(Hough,Point(x1,y1),Point(x2,y2),Scalar(255,0,0),2,LINE_AA);
    }
    imshow("HoughLine",Hough);

    vector<int> gray(256);

    int h=img.rows,w=img.cols;
    for(int row=0;row<h;++row) {
        for(int col=0;col<w;++col) {
            gray[img.at<uchar>(row,col)]++;
        }
    }

    int maxGray=0,maxPos=0;
    for(int i=0;i<256;++i) {
        if(gray[i]>maxGray) {
            maxGray=gray[i];
            maxPos=i;
        }
    }
    int argmax=0,maxPos2=0;
    for(int i=0;i<256;++i) {
        int cur=(i-maxPos)*(i-maxPos)*gray[i];
        if(cur>argmax) {
            argmax=cur;
            maxPos2=i;
        }
    }
    if(maxPos>maxPos2) swap(maxPos,maxPos2);
    int minGray=50000,minPos=0;
    for(int i=maxPos;i<=maxPos2;++i) {
        if(gray[i]<minGray) {
            minGray=gray[i];
            minPos=i;
        }
    }
    Mat histImage = Mat::zeros(Size(256,500),CV_8UC3);
    double total=50000;
    for(int i=0;i<256;++i) {
        Point last(i,500);
        Point now(cvRound(i), cvRound(500- cvRound(500*gray[i]/total)));
        if(i==maxPos||i==maxPos2) {
            line(histImage,last,now,Scalar(0,0,255),1,LINE_AA);
        }
        else {
            if(i!=minPos)
            line(histImage,last,now,Scalar(255,0,0),1,LINE_AA);
            else {
                line(histImage,last,now,Scalar(0,255,0),1,LINE_AA);

            }
        }
    }
    imshow("histImage",histImage);
    Mat binary;
    threshold(img,binary,minPos,255,THRESH_BINARY);
    imshow("Binary",binary);
    waitKey(0);
    return 0;
}
