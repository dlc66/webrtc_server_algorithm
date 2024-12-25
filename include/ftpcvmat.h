#ifndef FTPCVMAT_H
#define FTPCVMAT_H
#include <curl/curl.h>
#include <string>
#include <iostream>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/buffer.h>
#include <opencv2/opencv.hpp>
#include <vector>
class FtpCvmat
{
public:
    FtpCvmat(const std::string& url, const std::string& username, const std::string& password):
        url(url), username(username), password(password)
    {
        curl_global_init(CURL_GLOBAL_ALL);
        curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_USERPWD, (username + ":" + password).c_str());
            curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
            curl_easy_setopt(curl, CURLOPT_FTP_SSL, CURLFTPSSL_ALL);
            curl_easy_setopt(curl, CURLOPT_FTP_CREATE_MISSING_DIRS, 1L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        }
    }
    ~FtpCvmat(){
        if(curl)
        {
            curl_easy_cleanup(curl);
        }
        curl_global_cleanup();
    }

    int connect() {
        if (curl) {
            CURLcode res = curl_easy_perform(curl);
            return res == CURLE_OK;
        }
        return 0;
    }

    int download(const std::string& downloadPath) {
        if (curl) {
            file = fopen(downloadPath.c_str(), "wb");
            if (!file) {
                std::cerr << "Failed to open file for writing!" << std::endl;
                return 2;
            }
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);

            CURLcode res = curl_easy_perform(curl);
            fclose(file);

            if (res == CURLE_OK) {
                return 0; // 下载成功
            } else {
                std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
                return 1; // 下载失败
            }
        } else {
            return -1; // libcurl 初始化失败
        }
    }
    //fname 图片名 remoteFilePath 远程路径
    int uploadimage(const cv::Mat& image, const std::string& remoteFile) {
        if (curl) {
            std::vector<uchar> buffer;
            std::string::size_type idx = remoteFile.find_last_of(".");
            std::string extension = "";
           if (idx != std::string::npos) {
               extension = remoteFile.substr(idx);
           }
            bool success = cv::imencode(extension,image,buffer);
            if (!success)
            {
               std::cerr << "Failed to encode image!" << std::endl;
               return 2;
            }

//            std::string buffer = Mat2Base64(image,extension);
//            char* charPtr = const_cast<char*>(buffer.c_str());
            FILE* uploadFile = fmemopen(buffer.data(),buffer.size(),"rb");
            if (!uploadFile) {
                std::cerr << "Failed to open local file for reading!" << std::endl;
                return 3;
            }

            curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
            curl_easy_setopt(curl, CURLOPT_READFUNCTION, read_data);
            curl_easy_setopt(curl, CURLOPT_READDATA, uploadFile);
            curl_easy_setopt(curl, CURLOPT_URL, (url + remoteFile).c_str());

            CURLcode res = curl_easy_perform(curl);
            fclose(uploadFile);

            if (res == CURLE_OK) {
               return 0; // 上传成功
            } else {
                std::cerr << "Upload failed: " << curl_easy_strerror(res) << std::endl;
                return 1; // 上传失败
            }
        } else {
            return -1; // libcurl 初始化失败
        }
    }

private:
    CURL* curl;
    FILE* file;
    std::string url;
    std::string username;
    std::string password;
    static size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
        return fwrite(ptr, size, nmemb, stream);
    }

    static size_t read_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
        return fread(ptr, size, nmemb, stream);
    }
//    std::string Mat2Base64(const cv::Mat &img, std::string imgType) {
//        //Mat转base64
//        std::string img_data;
//        std::vector<uchar> vecImg;
//        std::vector<int> vecCompression_params;
//        vecCompression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
//        vecCompression_params.push_back(90);
//        imgType = "."+imgType;
//        cv::imencode(imgType, img, vecImg, vecCompression_params);
//        img_data = base64Encode(vecImg.data(), vecImg.size());
//        return img_data;
//    }
//    std::string base64Encode(const unsigned char* Data, int DataByte) {
//        // 编码表
//            const char EncodeTable[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

//            // 返回值
//            std::string strEncode;
//            unsigned char Tmp[4] = { 0 };
//            int LineLength = 0;

//            for (int i = 0; i < (int)(DataByte / 3); i++) {
//                Tmp[1] = *Data++;
//                Tmp[2] = *Data++;
//                Tmp[3] = *Data++;

//                strEncode += EncodeTable[Tmp[1] >> 2];
//                strEncode += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
//                strEncode += EncodeTable[((Tmp[2] & 0x0F) << 2) | ((Tmp[3] & 0xC0) >> 6)];
//                strEncode += EncodeTable[Tmp[3] & 0x3F];

//                LineLength += 4;
//                if (LineLength == 76) {
//                    strEncode += "\r\n";
//                    LineLength = 0;
//                }
//            }

//            // 对剩余数据进行编码
//            int Mod = DataByte % 3;
//            if (Mod == 1) {
//                Tmp[1] = *Data++;
//                strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
//                strEncode += EncodeTable[((Tmp[1] & 0x03) << 4)];
//                strEncode += "==";
//            } else if (Mod == 2) {
//                Tmp[1] = *Data++;
//                Tmp[2] = *Data++;
//                strEncode += EncodeTable[(Tmp[1] & 0xFC) >> 2];
//                strEncode += EncodeTable[((Tmp[1] & 0x03) << 4) | ((Tmp[2] & 0xF0) >> 4)];
//                strEncode += EncodeTable[((Tmp[2] & 0x0F) << 2)];
//                strEncode += "=";
//            }

//            return strEncode;
//    }
};

#endif // FTPCVMAT_H
