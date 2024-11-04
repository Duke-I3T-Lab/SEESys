//
// Created by slam on 8/29/23.
//

#include "UdpSender.h"
#include <iostream>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

UdpSender::UdpSender(const std::string& destIp, int destPort) {
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        std::cerr << "Error creating socket." << std::endl;
        return;
    }

    destAddr.sin_family = AF_INET;
    destAddr.sin_port = htons(destPort);
    destAddr.sin_addr.s_addr = inet_addr(destIp.c_str());
}

UdpSender::~UdpSender() {
    close(sockfd);
}

bool UdpSender::sendCSVRow(std::string csvString) {
    ssize_t sentBytes = sendto(sockfd, csvString.c_str(), csvString.size(), 0, (struct sockaddr*)&destAddr, sizeof(destAddr));
    if (sentBytes < 0) {
        //std::cerr << "Error sending data." << std::endl;
        return false;
    }
}
//
//bool UdpSender::sendCSVRow(int mnImCounter, double mdTimeStamp, int mnTrackMode,
//                           double mdBrightness, double mdContrast, double mdEntropy, double mdLaplacian,
//                           float mfMapPointAvgMinDepth, float mfMapPointVarMinDepth, float mnPrePOOutlier,
//                           int mnPrePOKeyMapLoss, int mnInlier, int mnPostPOOutlier, int mnPostPOKeyMapLoss,
//                           int mnMatchedInlier) {
//    std::ostringstream csvRow;
//    csvRow << doubleVar << "," << floatVar << "," << intVar;
//
//    std::string csvString = csvRow.str();
//
//    ssize_t sentBytes = sendto(sockfd, csvString.c_str(), csvString.size(), 0, (struct sockaddr*)&destAddr, sizeof(destAddr));
//    if (sentBytes < 0) {
//        std::cerr << "Error sending data." << std::endl;
//        return false;
//    }
//
//    return true;
//}