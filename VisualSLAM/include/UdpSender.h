//
// Created by slam on 8/29/23.
//

#ifndef ORB_SLAM3_UDPSENDER_H
#define ORB_SLAM3_UDPSENDER_H

#include <iostream>
#include <string>
#include <netinet/in.h>

class UdpSender {
public:
    UdpSender(const std::string& destIp, int destPort);
    ~UdpSender();

    bool sendCSVRow(std::string csvString);

private:
    int sockfd;
    struct sockaddr_in destAddr;
};


#endif