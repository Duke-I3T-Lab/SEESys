//
// Created by slam on 1/29/24.
//

// TCPClient.cpp
#include "TCPClient.h"
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

TCPClient::TCPClient(const std::string& serverAddress, int serverPort)
        : clientSocket(-1), serverAddress(serverAddress), serverPort(serverPort) {
}

TCPClient::~TCPClient() {
    Disconnect();
}

bool TCPClient::Connect() {
    clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket == -1) {
        std::cerr << "Socket creation failed." << std::endl;
        return false;
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(serverPort);
    serverAddr.sin_addr.s_addr = inet_addr(serverAddress.c_str());

    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1) {
        std::cerr << "Connection failed." << std::endl;
        close(clientSocket);
        return false;
    }

    return true;
}

bool TCPClient::SendMessage(const std::string& message) {
    if (send(clientSocket, message.c_str(), message.size(), 0) == -1) {
        std::cerr << "Error sending data." << std::endl;
        return false;
    }
    return true;
}

bool TCPClient::ReceiveMessage(std::string& receivedMessage) {
    char buffer[1024];
    int bytesReceived = recv(clientSocket, buffer, sizeof(buffer), 0);
    if (bytesReceived <= 0) {
        std::cerr << "Connection closed by the server." << std::endl;
        return false;
    }
    buffer[bytesReceived] = '\0';
    receivedMessage = buffer;
    return true;
}

void TCPClient::Disconnect() {
    if (clientSocket != -1) {
        close(clientSocket);
        clientSocket = -1;
    }
}

