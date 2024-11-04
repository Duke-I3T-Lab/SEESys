// include/udp_image_sender/UdpPointDistSender.h
#ifndef UDP_POINTDIST_SENDER_H
#define UDP_POINTDIST_SENDER_H

#include <string>
#include <opencv2/core/mat.hpp>

class UdpPointDistSender {
public:
    UdpPointDistSender(const std::string& destIp, int destPort);
    ~UdpPointDistSender();
    bool sendPointDist(const cv::Mat& pointDist, const std::string& format = ".png");

private:
    std::string destIp;
    int destPort;
};

#endif // UDP_POINTDIST_SENDER_H
