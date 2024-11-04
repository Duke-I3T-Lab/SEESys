//
// Created by slam on 1/29/24.
//

#ifndef ORB_SLAM3_TCPCLIENT_H
#define ORB_SLAM3_TCPCLIENT_H

#include <string>

class TCPClient {
public:
    TCPClient(const std::string& serverAddress, int serverPort);
    ~TCPClient();

    bool Connect();
    bool SendMessage(const std::string& message);
    bool ReceiveMessage(std::string& receivedMessage);
    void Disconnect();

private:
    int clientSocket;
    std::string serverAddress;
    int serverPort;
};

#endif ORB_SLAM3_TCPCLIENT_H

