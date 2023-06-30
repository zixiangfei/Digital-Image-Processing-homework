#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>

// fft变换后进行频谱搬移
void ffshift(cv::Mat &Img1) {
    //以下操作是移动图像（直流分量移动到中心）
    int cx = Img1.cols / 2;
    int cy = Img1.rows / 2;
    cv::Mat part1(Img1, cv::Rect(0, 0, cx, cy));
    cv::Mat part2(Img1, cv::Rect(cx, 0, cx, cy));
    cv::Mat part3(Img1, cv::Rect(0, cy, cx, cy));
    cv::Mat part4(Img1, cv::Rect(cx, cy, cx, cy));
    cv::Mat temp;
    part1.copyTo(temp);
    part4.copyTo(part1);
    temp.copyTo(part4);
    part2.copyTo(temp);
    part3.copyTo(part2);
    temp.copyTo(part3);
}

std::vector<std::vector<float> > mask8_0{
    std::vector<float> {1,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
};

std::vector<std::vector<float> > mask8_9{
    std::vector<float> {1,1,1,1,0,0,0,0},
    std::vector<float> {1,1,1,0,0,0,0,0},
    std::vector<float> {1,1,0,0,0,0,0,0},
    std::vector<float> {1,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
    std::vector<float> {0,0,0,0,0,0,0,0},
};

std::vector<std::vector<float> > mask8_35{
    std::vector<float> {1,1,1,1,1,1,1,1},
    std::vector<float> {1,1,1,1,1,1,1,0},
    std::vector<float> {1,1,1,1,1,1,0,0},
    std::vector<float> {1,1,1,1,1,0,0,0},
    std::vector<float> {1,1,1,1,0,0,0,0},
    std::vector<float> {1,1,1,0,0,0,0,0},
    std::vector<float> {1,1,0,0,0,0,0,0},
    std::vector<float> {1,0,0,0,0,0,0,0},
};

std::vector<std::vector<float> > mask8_53{
    std::vector<float> {1,1,1,1,1,1,1,1},
    std::vector<float> {1,1,1,1,1,1,1,1},
    std::vector<float> {1,1,1,1,1,1,1,1},
    std::vector<float> {1,1,1,1,1,1,1,1},
    std::vector<float> {1,1,1,1,1,1,1,0},
    std::vector<float> {1,1,1,1,1,1,0,0},
    std::vector<float> {1,1,1,1,1,0,0,0},
    std::vector<float> {1,1,1,1,0,0,0,0},
};

std::vector<std::vector<float> > mask16_0{
        std::vector<float> {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
};

std::vector<std::vector<float> > mask16_9{
        std::vector<float> {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
};

std::vector<std::vector<float> > mask16_35{
        std::vector<float> {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
};

std::vector<std::vector<float> > mask16_135{
        std::vector<float> {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        std::vector<float> {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0},
        std::vector<float> {1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0},
        std::vector<float> {1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0},
        std::vector<float> {1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        std::vector<float> {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
};

void homework1(cv::Mat img,int D1,int D2) {
    img.convertTo(img,CV_32F);//转换成浮点类型
    cv::Mat complexImg;
    cv::dft(img,complexImg,CV_HAL_DFT_COMPLEX_OUTPUT); //离散傅立叶变换
    std::vector<cv::Mat> vecImg;
    cv::split(complexImg,vecImg); //将图像实部和虚部两个部分分开
    ffshift(vecImg[0]); //分别对两个通道进行频谱转移
    ffshift(vecImg[1]);
    cv::Mat mag,magLog,magNorm;
    cv::magnitude(vecImg[0],vecImg[1],mag); //计算图像幅值
    mag += cv::Scalar::all(1);
    cv::log(mag, magLog); //幅值对数化log（1+m）便于观察频谱信息
    cv::normalize(magLog, magNorm, 1, 0, cv::NORM_MINMAX); //归一化
    cv::imshow("1",magNorm);
    //std::cout<<magNorm.rows<<" "<<magNorm.cols<<'\n';

    int height = mag.rows, width = mag.cols;
    //std::cout<<mag.rows<<" "<<mag.cols<<'\n';
    int cx = height / 2, cy = width / 2;
    cv::Mat lowerFilter0 = cv::Mat::zeros(cv::Size(width, height),CV_32F);
    cv::Mat lowerFilter1 = cv::Mat::zeros(cv::Size(width, height),CV_32F);
    std::vector<cv::Mat> lower;
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            if((row - cx)*(row - cx)+(col - cy)*(col - cy)<(float)(D1*D1)) {
                //std::cout<<row<<" "<<col<<'\n';
                lowerFilter0.at<float>(row, col) = vecImg[0].at<float>(row, col);
                lowerFilter1.at<float>(row, col) = vecImg[1].at<float>(row, col);
            }
        }
    }
    lower.push_back(lowerFilter0);
    lower.push_back(lowerFilter1);
    cv::Mat lowerMag;
    cv::magnitude(lowerFilter0, lowerFilter1, lowerMag);
    lowerMag += cv::Scalar::all(1);
    cv::log(lowerMag,lowerMag);
    cv::normalize(lowerMag, lowerMag, 1, 0, cv::NORM_MINMAX); //归一化
    cv::imshow("2",lowerMag);

    ffshift(lowerFilter0);
    ffshift(lowerFilter1);
    lower.clear();
    lower.push_back(lowerFilter0);
    lower.push_back(lowerFilter1);

    cv::Mat lowerMerge;
    cv::merge(lower, lowerMerge);
    cv::idft(lowerMerge, lowerMerge);
    cv::split(lowerMerge, lower);
    cv::Mat lowerRes;
    cv::magnitude(lower[0], lower[1], lowerRes);
    cv::normalize(lowerRes, lowerRes, 1, 0, cv::NORM_MINMAX);
    cv::imshow("3",lowerRes);

    cv::Mat highFilter0 = cv::Mat::zeros(cv::Size(width, height), CV_32F);
    cv::Mat highFilter1 = cv::Mat::zeros(cv::Size(width, height), CV_32F);
    std::vector<cv::Mat> high;
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            if((row - cx)*(row - cx)+(col - cy)*(col - cy)>(float)(D2 * D2)) {
                highFilter0.at<float>(row, col) = vecImg[0].at<float>(row, col);
                highFilter1.at<float>(row, col) = vecImg[1].at<float>(row, col);
            }
        }
    }
    high.push_back(highFilter0);
    high.push_back(highFilter1);
    cv::Mat highMag;
    cv::magnitude(highFilter0, highFilter1, highMag);
    highMag += cv::Scalar::all(1);
    cv::log(highMag, highMag);
    cv::normalize(highMag, highMag, 1, 0, cv::NORM_MINMAX);
    cv::imshow("4",highMag);

    ffshift(highFilter0);
    ffshift(highFilter1);
    high.clear();
    high.push_back(highFilter0);
    high.push_back(highFilter1);

    cv::Mat highMerge;
    cv::merge(high, highMerge);
    cv::idft(highMerge, highMerge);
    cv::split(highMerge, high);
    cv::Mat highRes;
    cv::magnitude(high[0], high[1], highRes);
    cv::normalize(highRes, highRes, 1, 0, cv::NORM_MINMAX);
    cv::imshow("5",highRes);

    cv::waitKey(0);
}

void dctFilter(cv::Mat src, std::vector<std::vector<float>> mask, int size) {
    cv::Mat res = cv::Mat::zeros(src.size(),CV_32F);
    int height = src.rows, width = src.cols;
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            res.at<float>(row, col) = src.at<float>(row, col) * mask[row % size][col % size];
        }
    }

    cv::imshow("test",res);
    for(int row = 0; row < height; row += size) {
        for(int col = 0; col < width; col += size) {
            cv::Rect rect = cv::Rect(col, row, size, size);
            cv::idct(res(rect), res(rect));
        }
    }
    cv::normalize(res, res, 1, 0, cv::NORM_MINMAX);
    cv::imshow("test2", res);
}

void homework2(cv::Mat src) {
    int height = src.rows, width = src.cols;
    cv::Mat src8x8,src16x16;
    src.copyTo(src8x8);
    src.copyTo(src16x16);
    //std::cout<<height<<" "<<width<<'\n';
    for(int row = 0; row < height; row += 8) {
        for(int col = 0; col < width; col += 8) {
            if(row + 8 > height || col + 8 > width) continue;
            cv::Rect rect = cv::Rect(col, row, 8 ,8);
            //std::cout<<row<<" "<<col<<'\n';
            cv::dct(src8x8(rect),src8x8(rect));
        }
    }
    for(int row = 0; row < height; row += 16) {
        for(int col = 0; col < width; col += 16) {
            if(row + 16 > height || col + 16 > width) continue;
            cv::Rect rect = cv::Rect(col, row, 16, 16);
            //std::cout<<row<<" "<<col<<'\n';
            cv::dct(src16x16(rect), src16x16(rect));
        }
    }
    //cv::normalize(src16x16, src16x16, 1, 0, cv::NORM_MINMAX);
    //cv::imshow("test",src16x16);
    //cv::imwrite("dct8x8.jpeg",src8x8);
    //cv::imwrite("dct16x16.jpeg",src16x16);
    //dctFilter(src8x8, mask8_9, 8);
    dctFilter(src16x16, mask16_35, 16);
    cv::waitKey(0);
};

int main() {
    std::string home="/Users/zhengluzhou/Desktop/study/Digital Image Processing/experience/experience_3/"; //工作目录
    cv::Mat src = cv::imread(home+"experience3.jpg");
    cv::Mat gray = cv::Mat::zeros(src.size(), src.type());
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
//    homework1(gray,100,200);
    cv::Mat src2 = cv::imread(home+"experience3_2.jpg");
    cv::Mat gary2 = cv::Mat::zeros(src2.size(), src2.type());
    cv::cvtColor(src2, gary2, cv::COLOR_BGR2GRAY);
    gary2.convertTo(gary2,CV_32F);
    cv::Mat reshapeGary2  = cv::Mat::zeros(cv::Size(720, 1120), CV_32F);
    for(int height = 0; height < 1120; ++height) {
        for(int width = 0; width < gary2.cols; ++width) {
            reshapeGary2.at<float>(height, width) = gary2.at<float>(height, width);
        }
    }
    homework2(reshapeGary2);
    cv::waitKey(0);
    return 0;
}
