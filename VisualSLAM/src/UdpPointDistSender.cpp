// src/UdpPointDistSender.cpp
#include "UdpPointDistSender.h"
#include <boost/asio.hpp>
#include <opencv2/opencv.hpp>

using boost::asio::ip::udp;
using namespace cv;

UdpPointDistSender::UdpPointDistSender(const std::string& ip, int port) : destIp(ip), destPort(port) {}

UdpPointDistSender::~UdpPointDistSender() {}

bool UdpPointDistSender::sendPointDist(const cv::Mat& pointDist, const std::string& format) {
    try {
        boost::asio::io_service io_service;
        udp::resolver resolver(io_service);
        udp::resolver::query query(udp::v4(), destIp, std::to_string(destPort));
        udp::endpoint receiver_endpoint = *resolver.resolve(query);
        udp::socket socket(io_service);
        socket.open(udp::v4());

        std::vector<uchar> buf;
        if (!imencode(format, pointDist, buf)) {
            return false;
        }

        socket.send_to(boost::asio::buffer(buf), receiver_endpoint);
        return true;
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return false;
    }
}

// examples/main.cpp
// #include "udp_image_sender/UdpImageSender.h"
// #include <opencv2/opencv.hpp>

// int main() {
//     cv::Mat image = cv::imread("path_to_your_image.png", cv::IMREAD_COLOR);
//     if (image.empty()) {
//         std::cerr << "Error loading image." << std::endl;
//         return -1;
//     }

//     UdpImageSender sender("127.0.0.1", 12345);
//     if (!sender.sendImage(image)) {
//         std::cerr << "Failed to send image." << std::endl;
//     }

//     return 0;
// }
