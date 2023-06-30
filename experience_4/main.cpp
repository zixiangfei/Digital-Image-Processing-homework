#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <bitset>
#include <queue>
#include <functional>
#include <vector>

std::string home = "/Users/zhengluzhou/Desktop/study/Digital Image Processing/experience/experience_4/";

struct RLEcoding {
    uchar color;
    ushort value;
};

void calculateRLE(std::vector<uchar> colors, int height, int width) {
    int length = colors.size();
    //std::cout<<length<<'\n';
    uchar lastColor = 0;
    ushort value = 0;
    std::vector<RLEcoding> res;
    for(int idx = 0; idx < length; ++ idx) {
        if(idx == 0) {
            lastColor = colors[idx];
            value++;
        }
        else {
            if(lastColor != colors[idx]) {
                res.push_back({lastColor, value});
                lastColor = colors[idx];
                value = 1;
            }
            else {
                value++;
            }
        }
    }
    res.push_back({lastColor, value});
    ushort maxLen = 0;
    for(int idx = 0; idx < res.size(); ++idx) {
        maxLen = std::max(maxLen, res[idx].value);
    }
    int len = ceil(log2(maxLen));
    double resultLen = res.size()*(8 + len);
    double originLen = height * width * 8;
    double radio = resultLen / originLen * 100;
    printf("图像压缩比：%lf%%\n",radio);
}

void RLE1(cv::Mat src) {
    int height = src.rows, width = src.cols;
    std::cout<<height<<" "<<width<<'\n';
    int row = 0, col = 0, times = 0;
    std::vector<uchar> colors;
    while(row < height && col < width) {
        times %= 2;
        if(times) {
            for(int i = col, j = row; i >= row && j <= col; i--, j++) colors.push_back(src.at<uchar>(i, j));
        }
        else {
            for(int i = row, j = col; i <= col && j >= row; i++, j--) colors.push_back(src.at<uchar>(i, j));
        }
        if(col == width - 1) {
            row++;
        }
        else {
            col++;
        }
        times++;
    }
    int length = colors.size();
//    std::cout<<length<<"\n";
//    for(int i = 0; i < length; ++i) {
//        std::cout<<(int) colors[i]<<' ';
//    }
//    std::cout<<'\n';
    calculateRLE(colors, height, width);
}

void RLE2(cv::Mat src) {
    int height = src.rows, width = src.cols;
    std::cout<<height<<" "<<width<<'\n';
    int row = 0, col = 0 ,times = 0;
    std::vector<uchar> colors;
    colors.push_back(src.at<uchar>(col, row));
    while(row < height && col < width) {
        //std::cout<<row<<" "<<col<<'\n';
        times %= 2;
        if(times) {
            row++;
            if(row >= height || col >= width) break;
            while(row != col) {
                //std::cout<<row<<" "<<col<<'\n';
                colors.push_back(src.at<uchar>(col, row));
                col++;
            }
            while(row > 0) {
                //std::cout << row << " " << col << '\n';
                colors.push_back(src.at<uchar>(col, row));
                row--;
            }
            colors.push_back(src.at<uchar>(col, row));
            times++;
        }
        else {
            col++;
            if(row >= height || col >= width) break;
            //std::cout<<row<<" "<<col<<'\n';
            while(row != col) {
                //std::cout<<row<<" "<<col<<'\n';
                colors.push_back(src.at<uchar>(col, row));
                row++;
            }
            while(col > 0) {
                //std::cout<<row<<" "<<col<<'\n';
                colors.push_back(src.at<uchar>(col, row));
                col--;
            }
            colors.push_back(src.at<uchar>(col, row));
            times++;
        }
    }
    calculateRLE(colors, height, width);
}

struct BinaryNode {
    uchar color;
    int weight;
    struct BinaryNode* lChild;
    struct BinaryNode* rChild;
    std::string code;
};

class cmp{
public:
    bool operator()(BinaryNode *a, BinaryNode *b) {
        return a->weight > b->weight;
    }
};

struct HuffmanCode {
    uchar color;
    std::string code;
    int weight;
};

void dfs(BinaryNode *root, std::string code, std::vector<HuffmanCode> &coding) {
    //std::cout<<(int)root->weight<<" "<<code<<'\n';
    if(!root) return;
    if(root->lChild == nullptr && root->rChild == nullptr) {
        coding.push_back({root->color, code, root->weight});
        return;
    }
    if(root->lChild) {
        dfs(root->lChild, code+"0", coding);
    }
    if(root->rChild) {
        dfs(root->rChild, code+"1", coding);
    }

}

void build(cv::Mat src, std::vector<std::vector<HuffmanCode>> &res) {
    int height = src.rows, width = src.cols;
    //std::cout<<height<<" "<<width<<'\n';
    uchar colors[256]={0};
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            colors[src.at<uchar>(row, col)]++;
        }
    }
    std::priority_queue<BinaryNode*,std::vector<BinaryNode *>, cmp> queue;
    for(int i = 0; i < 256; ++i) {
        if(!colors[i]) continue;
        BinaryNode* curColor = (BinaryNode *)malloc(sizeof(BinaryNode));
        curColor->weight = colors[i];
        curColor->color = i;
        curColor->lChild = nullptr;
        curColor->rChild = nullptr;
        //std::cout<<(int)curColor->color<<" "<<(int)curColor->weight<<'\n';
        queue.push(curColor);
    }
    while(queue.size() > 1) {
        BinaryNode *left, *right;
        left = queue.top();
        queue.pop();
        right = queue.top();
        queue.pop();
        //std::cout<<"left: "<<(int)left->color<<" "<<(int)left->weight<<'\n';
        //std::cout<<"right: "<<(int)right->color<<" "<<(int)right->weight<<'\n';
        BinaryNode* father = (BinaryNode *) malloc(sizeof(BinaryNode));
        father->weight = left->weight + right->weight;
        father->color = 0;
        father->lChild = left;
        father->rChild = right;
        queue.push(father);
    }
    BinaryNode* root = queue.top();
    queue.pop();
    std::string code="";
    std::vector<HuffmanCode> coding;
    dfs(root, code, coding);
//    std::cout<<coding.size()<<'\n';
//    for(int i = 0; i < coding.size(); ++i) {
//        std::cout<<(int)coding[i].color<<" "<<coding[i].code<<" "<<coding[i].weight<<'\n';
//    }
    res.push_back(coding);
}

void calculateHuffman(std::vector<std::vector<HuffmanCode>> res) {
    double originLen = 0;
    double resultLen = 0;
    for(int i = 0; i < res.size(); ++ i) {
        for(int j = 0; j < res[i].size(); ++ j) {
            originLen += res[i][j].weight * 8;
            resultLen += res[i][j].weight * res[i][j].code.size();
        }
    }
    double radio = resultLen / originLen * 100;
    printf("图像压缩比：%lf%%\n",radio);
}

void huffmanCoding(cv::Mat src) {
    std::vector<std::vector<HuffmanCode> >res;
    int height = src.rows, width = src.cols;
    for(int row = 0; row < height; row += 8) {
        for(int col = 0; col < width; col += 8) {
            cv::Rect rect = cv::Rect(col, row, std::min(8, width - col), std::min(8, height - row));
            build(src(rect),res);
        }
    };
    calculateHuffman(res);
}

void homework1() {
    cv::Mat src = cv::imread(home+"experience44.JPG"); //目标图像
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY); //转化为灰度图像
    cv::Mat reshapeGray = cv::Mat::zeros(cv::Size(711, 711), CV_8U); //转换成方阵
    int height = gray.rows, width = gray.cols;
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++ col) {
            reshapeGray.at<uchar>(row, col) = gray.at<uchar>(row, col);
        }
    }
//    cv::Mat test = cv::Mat::zeros(cv::Size(5, 5), CV_8U); //测试用矩阵
//    for(int i = 0; i < 5; ++i) {
//        for(int j = 0; j < 5; ++j) {
//            test.at<uchar>(j, i) = i * 5 + j;
//        }
//    }
//    RLE1(reshapeGray);//z形行程编码
    //RLE2(reshapeGray);//回形行程编码
    cv::Mat huffmanSrc = cv::imread(home+"experience4.JPG");
    std::cout<<huffmanSrc.rows<<" "<<huffmanSrc.cols<<'\n';
    huffmanCoding(src);//霍夫曼编码
    //cv::imshow("gray",gray);
    //cv::waitKey(0);
}

std::vector<std::vector<float>> Q_y = {
        std::vector<float>{16, 11, 10, 16, 24, 40, 51, 61},
        std::vector<float>{12, 12, 14, 19, 26, 58, 60, 55},
        std::vector<float>{14, 13, 16, 24, 40, 57, 69, 56},
        std::vector<float>{14, 17, 22, 29, 51, 87, 80, 62},
        std::vector<float>{18, 22, 37, 56, 68, 109, 103, 77},
        std::vector<float>{24, 35, 55, 64, 81, 104, 113, 92},
        std::vector<float>{49, 64, 78, 87, 103, 121, 120, 101},
        std::vector<float>{72, 92, 95, 98, 112, 100, 103, 99},
};//标准亮度量化表

std::vector<std::vector<float>> Q_c= {
        std::vector<float>{17, 18, 24, 47, 99, 99, 99, 99},
        std::vector<float>{18, 21, 26, 66, 99, 99, 99, 99},
        std::vector<float>{24, 26, 56, 99, 99, 99, 99, 99},
        std::vector<float>{47, 66, 99, 99, 99, 99, 99, 99},
        std::vector<float>{99, 99, 99, 99, 99, 99, 99, 99},
        std::vector<float>{99, 99, 99, 99, 99, 99, 99, 99},
        std::vector<float>{99, 99, 99, 99, 99, 99, 99, 99},
        std::vector<float>{99, 99, 99, 99, 99, 99, 99, 99},
};//标准色差量化表

void quantization(cv::Mat src, std::vector<std::vector<float>> Q, int quality) {
    int height = src.rows, width = src.cols;
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            double Q_now;
            if(quality < 100 && quality >= 50) {
                Q_now = std::max(cvRound((2.0 - quality / 50.0) * Q[col][row]), 1);
            }
            else {
                Q_now = cvRound(50 / quality * Q[row][col]);
            }
            //std::cout<<Q_now<<'\n';
            src.at<float>(col, row) = cvRound(src.at<float>(col, row) / Q_now);
        }
    }
}

struct floatRLEcoding {
    float color;
    ushort value;
};

struct jpegRLEcoding {
    std::vector<floatRLEcoding> coding;
    int col, row, width, height;
    int maxLen;
};

jpegRLEcoding jpegRLE(cv::Mat src, int posX, int posY) {
    jpegRLEcoding ret;
    ret.col = posX;
    ret.row = posY;
    int height = src.rows, width = src.cols;
    ret.width = width, ret.height = height;
    int row = 0, col = 0, times = 0;
    std::vector<float> colors;
    while(row < height && col < width) {
        //std::cout<<row<<" "<<col<<" "<<width<<" "<<height<<'\n';
        times %= 2;
        if(times) {
            for(int i = col, j = row; i >= row && j <= col; i--, j++) colors.push_back(src.at<float>(i, j));
        }
        else {
            for(int i = row, j = col; i <= col && j >= row; i++, j--) colors.push_back(src.at<float>(i, j));
        }
        if(col == width - 1) {
            row++;
        }
        else {
            col++;
        }
        times++;
    }
    int length = colors.size();
    //std::cout<<length<<'\n';
    float lastColor = 0;
    ushort value = 0;
    std::vector<floatRLEcoding> res;
    for(int idx = 0; idx < length; ++ idx) {
        if(idx == 0) {
            lastColor = colors[idx];
            value++;
        }
        else {
            if(lastColor != colors[idx]) {
                res.push_back({lastColor, value});
                lastColor = colors[idx];
                value = 1;
            }
            else {
                value++;
            }
        }
    }
    res.push_back({lastColor, value});
    ushort maxLen = 0;
    for(int idx = 0; idx < res.size(); ++idx) {
        maxLen = std::max(maxLen, res[idx].value);
    }
    maxLen = ceil(log((double)maxLen) / log(2.0));
    ret.maxLen = maxLen;
    ret.coding = res;
    return ret;
}

void decodingRLE(cv::Mat src, std::vector<floatRLEcoding> res) {
    int height = src.rows, width = src.cols;
    int length = res.size();
    std::vector<float> colors;
    //std::cout<<length<<'\n';
    for(int idx = 0; idx < length; ++idx) {
        ushort cnt = res[idx].value;
        while(cnt--) {
            colors.push_back(res[idx].color);
        }
    }
    //std::cout<<colors.size()<<'\n';
    int row = 0, col = 0, times = 0, num = 0;
    while(row < height && col < width) {
        //std::cout<<row<<" "<<col<<" "<<width<<" "<<height<<'\n';
        times %= 2;
        if(times) {
            for(int i = col, j = row; i >= row && j <= col; i--, j++) {
                src.at<float>(i, j) = colors[num++];
            }
        }
        else {
            for(int i = row, j = col; i <= col && j >= row; i++, j--) {
                src.at<float>(i, j) = colors[num++];
            }
        }
        if(col == width - 1) {
            row++;
        }
        else {
            col++;
        }
        times++;
    }
}

void iQuantization(cv::Mat src, std::vector<std::vector<float>> Q, int quality) {
    int height = src.rows, width = src.cols;
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            double Q_now;
            if(quality <= 100 && quality >= 50) {
                Q_now = std::max(cvRound((2.0 - quality / 50.0) * Q[col][row]), 1);
            }
            else {
                Q_now = cvRound(50 / quality * Q[row][col]);
            }
            src.at<float>(col, row) = (float)cvRound(src.at<float>(col, row) * Q_now);
        }
    }
}

void calculateJpeg(cv::Mat src, cv::Mat dst, std::vector<jpegRLEcoding> channels1, std::vector<jpegRLEcoding> channels2, std::vector<jpegRLEcoding> channels3) {
    int height = src.rows, width = src.cols;
    long long src_size = height * width * 24;
    long long dst_size = 0;
    for(int idx = 0; idx < channels1.size(); ++ idx) {
        dst_size += 64 * channels1[idx].maxLen;
        dst_size += 64 * channels2[idx].maxLen;
        dst_size += 64 * channels3[idx].maxLen;
        //std::cout<<channels1[idx].maxLen<<" "<<channels2[idx].maxLen<<" "<<channels3[idx].maxLen<<'\n';
    }
    double radio = (double)dst_size / (double)src_size;
    double RMSE = 0;
    double MSE = 0;
    for(int row = 0; row < height; ++row) {
        for(int col = 0; col < width; ++col) {
            double b1 = src.at<cv::Vec3b>(col, row)[0], b0 = dst.at<cv::Vec3b>(col, row)[0];
            double g1 = src.at<cv::Vec3b>(col, row)[1], g0 = dst.at<cv::Vec3b>(col, row)[1];
            double r1 = src.at<cv::Vec3b>(col, row)[2], r0 = dst.at<cv::Vec3b>(col, row)[2];
            MSE += (b1 - b0) * (b1 - b0) + (g1 - g0) * (g1 - g0) + (r1 - r0) * (r1 - r0);
        }
    }
    MSE /= (double)(height * width) * 3;
    RMSE = sqrt(MSE);
    printf("src_size: %lld, dst_size: %lld, radio = %lf%%, RMSE = %lf.\n", src_size, dst_size, radio * 100.0, RMSE);
}

void homework2(int quality) {
    cv::Mat src = cv::imread(home+"experience4_1.JPG");
    cv::imshow("src", src);
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> test;
    cv::split(hsv, test);
    hsv.convertTo(hsv, CV_32F);


    //编码
    std::vector<cv::Mat> channels;
    std::vector<jpegRLEcoding> channels1;
    std::vector<jpegRLEcoding> channels2;
    std::vector<jpegRLEcoding> channels3;
    cv::split(hsv, channels);
    int height = hsv.rows, width = hsv.cols;
    int cnt = 0;
    for(int row = 0; row < height; row += 8) {
        for(int col = 0; col < width; col += 8) { //分块
            cnt++;
            cv::Rect rect = cv::Rect(col, row, 8, 8);
            //通道1
            cv::dct(channels[0](rect), channels[0](rect));// dct
            quantization(channels[0](rect),Q_c, quality); //量化
            channels1.push_back(jpegRLE(channels[0](rect), col, row));//z字形编码
            //通道2
            cv::dct(channels[1](rect), channels[1](rect));
            quantization(channels[1](rect), Q_c, quality);
            channels2.push_back(jpegRLE(channels[1](rect), col, row));
            //通道3
            cv::dct(channels[2](rect), channels[2](rect));
            quantization(channels[2](rect), Q_y, quality);
            channels3.push_back(jpegRLE(channels[2](rect), col, row));
        }
    }
    std::vector<cv::Mat> decode;
    cv::Mat decode1 = cv::Mat::zeros(cv::Size(width, height), CV_32F);
    cv::Mat decode2 = cv::Mat::zeros(cv::Size(width, height), CV_32F);
    cv::Mat decode3 = cv::Mat::zeros(cv::Size(width, height), CV_32F);
    int channelLength = channels1.size();
    for(int idx = 0; idx < channelLength; ++idx) { //解码
        jpegRLEcoding curCoding1 = channels1[idx];
        jpegRLEcoding curCoding2 = channels2[idx];
        jpegRLEcoding curCoding3 = channels3[idx];
        int col = curCoding1.col, row = curCoding1.row, curWidth = curCoding1.width, curHeight = curCoding1.height;
        cv::Rect rect = cv::Rect(col, row, curWidth, curHeight);
        decodingRLE(decode1(rect), curCoding1.coding);
        decodingRLE(decode2(rect), curCoding2.coding);
        decodingRLE(decode3(rect), curCoding3.coding);
    }
    for(int row = 0; row < height; row += 8) {
        for(int col = 0; col < width; col += 8) {
            cv::Rect rect = cv::Rect(col, row, 8, 8);
            //通道一
            iQuantization(decode1(rect), Q_c, quality);
            cv::idct(decode1(rect), decode1(rect));
            //通道二
            iQuantization(decode2(rect), Q_c, quality);
            cv::idct(decode2(rect), decode2(rect));
            //通道三
            iQuantization(decode3(rect), Q_y, quality);
            cv::idct(decode3(rect), decode3(rect));
        }
    }
    decode1.convertTo(decode1, CV_8U);
    decode2.convertTo(decode2, CV_8U);
    decode3.convertTo(decode3, CV_8U);
    decode.push_back(decode1);
    decode.push_back(decode2);
    decode.push_back(decode3);
    cv::Mat dst;
    cv::merge(decode, dst);
    cv::cvtColor(dst, dst, cv::COLOR_HSV2BGR);
    calculateJpeg(src, dst, channels1, channels2, channels3);
    cv::imshow("dst", dst);
    cv::waitKey(0);
};

int main() {
    //homework1();
    //freopen("out.txt","w",stdout);
    int quality = 80;
    homework2(quality);
    return 0;
}
