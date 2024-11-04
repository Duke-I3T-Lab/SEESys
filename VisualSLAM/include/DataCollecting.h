//
// Created by tianyi on 6/15/23.
//

#ifndef ORB_SLAM3_DATACOLLECTING_H
#define ORB_SLAM3_DATACOLLECTING_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <deque>
#include <fstream>


#include <iomanip>
#include <ctime>

#include <thread>
#include <cmath>
#include <numeric>
#include <opencv2/core/core.hpp>

#include "Tracking.h"
#include "LocalMapping.h"
#include "LoopClosing.h"

#include "Frame.h"
#include "KeyFrame.h"
#include "sophus/se3.hpp"

#include <mutex>
#include <chrono>

#include "UdpSender.h"
#include "TCPClient.h"
#include "UdpPointDistSender.h"


namespace  ORB_SLAM3 {
    class System;

    class Tracking;

    class LocalMapping;

    class LoopClosing;

    class FrameBuffer {
    public:
        FrameBuffer(size_t capacity = 30) : capacity_(capacity) {}

        void insert(const cv::Mat &frame) {
            if (buffer_.size() >= capacity_) {
                buffer_.pop_front();
            }
            buffer_.push_back(frame.clone());
        }

        void reset() {
            buffer_.clear();
        }

        void saveSnapshot(const std::string &filename) {
            if (!isValid()) {
                return; // Do nothing if buffer is not valid
            }

            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << filename << std::endl;
                return;
            }

            // Frames to save: first, 10th, 20th, and last
            std::vector<cv::Mat> framesToSave = { buffer_.front(), buffer_[9], buffer_[19], buffer_.back() };

            for (const cv::Mat &frame : framesToSave) {
                // Save the size of the frame (rows, cols, type)
                int rows = frame.rows, cols = frame.cols, type = frame.type();
                file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
                file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
                file.write(reinterpret_cast<const char*>(&type), sizeof(type));

                // Save the raw frame data
                file.write(reinterpret_cast<const char*>(frame.data), frame.total() * frame.elemSize());
            }

            file.close();
        }

        bool isValid() const {
            return buffer_.size() == capacity_;
        }

    private:
        std::deque<cv::Mat> buffer_;
        size_t capacity_;

        cv::Mat mergeFrames(const std::vector<cv::Mat> &frames) {
            // Implement logic to merge frames into a 3D cv::Mat
            // This is a placeholder for the actual implementation
            return cv::Mat(); // Replace with actual merged Mat
        }
    };

    class DataCollecting {
    public:
        DataCollecting(System *pSys, Atlas *pAtlas, const float bMonocular, bool bInertial,
                       const string &_strSeqName = std::string());

        void SetTracker(Tracking *pTracker);

        void SetLocalMapper(LocalMapping *pLocalMapper);

        void SetLoopCloser(LoopClosing *pLoopCloser);

        void InitializeDataCollector();

        void InitializeUDPTSender();

        void InitializeUDPPointDistSender();

        // Main function
        void Run();

        void RequestFinish();

        // Public functions to collect data
        void CollectImageTimeStamp(const double &timestamp);

        void CollectImageFileName(string &filename);

        void CollectImagePixel(cv::Mat &imGrey);

        void CollectDistribution(cv::Mat &distMatrix);

        void CollectCurrentFrame(const Frame &frame);

        void CollectTotalNumKeyFrames(const int totalNumKF);

        void CollectCurrentFrameTrackMode(const int &nTrackMode);

        void CollectCurrentFrameTrackMethod(const int &nTrackMethod);

        void CollectCurrentFramePrePOOutlier(const int &nPrePOOutlier);

        void CollectCurrentFramePrePOKeyMapLoss(const int &nPrePOKeyMapLoss);

        void CollectCurrentFrameInlier(const int &nInlier);

        void CollectCurrentFramePostPOOutlier(const int &nPostPOOutlier);

        void CollectCurrentFramePostPOKeyMapLoss(const int &nPostPOKeyMapLoss);

        void CollectCurrentFrameMatchedInlier(const int &nMatchedInlier);

        void CollectCurrentFrameMapPointDepth(const Frame &frame);
        //void CollectCurrentFrame;
        //void CollectCurrentKeyFrame(KeyFrame currentKeyFrame);

        void CollectLocalMappingNewKFinQueue(const int num_NewKFinQueue);

        void CollectLocalMappingBANumber(const int num_FixedKF_BA, const int num_OptKF_BA,
                                         const int num_MPs_BA, const int num_edges_BA);

        void CollectLocalMappingBAOptimizer(const float fLocalBAVisualError);

        void CollectLoopClosureBAOptimizer(const float fGlobalBAVisualError);

        void CollectLoopClosureNewKFinQueue(const int num_NewKFinQueue);

        float GetActionThRefRatio();
        int GetActionMaxFrames();
        int GetActionMinFrames();

    protected:

        System *mpSystem;
        Atlas *mpAtlas;
        bool mbMonocular;
        bool mbInertial;

        bool mbFinished;

        // member pointers to the three main modules
        Tracking *mpTracker;
        LocalMapping *mpLocalMapper;
        LoopClosing *mpLoopCloser;

        void CollectCurrentTime();

        void InitializeCSVLogger();

        void WriteRowCSVLogger();

        std::string msSeqName;
        std::string msCurrentTime;

        // Declare the mFileLogger as an reference using the & operator
        // So that I can copy the actual ofstream to it.
        // Settings of the .csv file
        std::ofstream mFileLogger;
        std::string msCSVFileName;

        UdpSender *mpUdpSender;
        UdpPointDistSender *mpUdpPointDistSender;

        void SendRowUDP();
        void SendPointDistUDP(const cv::Mat& pointDist);

        std::vector<std::string> mvsColumnFeatureNames = {"Counter", "TimeStamp", "TrackMode", "TrackMethod", \
                                                   "Brightness", "Contrast", "Entropy", "Laplacian", \
                                                   "AvgMPDepth", "VarMPDepth", \
                                                   "PrePOOutlier", "PrePOKeyMapLoss", \
                                                   "Inlier", "PostPOOutlier", "PostPOKeyMapLoss", "MatchedInlier", \
                                                   "NumberKeyPoints", "PX", "PY", "PZ", "QX", "QY", "QZ", "QW", \
                                                   "DX", "DY", "DZ", "Yaw", "Pitch", "Roll", \
                                                   "num_LocalKFinQueue", "num_FixedKF_BA", "num_OptKF_BA", "num_MPs_BA",
                                                          "num_edges_BA", \
                                                   "num_local_BA", "local_visual_BA_Err", "num_GlobalKFinQueue",
                                                          "num_Global_BA", "global_visual_BA_Err",
                                                          "CollectionLatency", "RefRatio", "MinFrame", "MaxFrame", "TotalNumKF"};

        // Mutexs for locks
        std::timed_mutex mMutexNewFrameProcessed;

        std::timed_mutex mMutexImageTimeStamp;
        std::timed_mutex mMutexImageFileName;
        std::timed_mutex mMutexImageCounter;
        std::timed_mutex mMutexImagePixel;
        std::timed_mutex mMutexImageFeatures;

        std::timed_mutex mMutexCurrentFrame;
        std::timed_mutex mMutexCurrentFrameTrackMode;
        std::timed_mutex mMutexCurrentFrameTrackMethod;
        std::timed_mutex mMutexCurrentFramePrePOOutlier;
        std::timed_mutex mMutexCurrentFramePrePOKeyMapLoss;
        std::timed_mutex mMutexCurrentFrameInlier;
        std::timed_mutex mMutexCurrentFramePostPOOutlier;
        std::timed_mutex mMutexCurrentFramePostPOKeyMapLoss;
        std::timed_mutex mMutexCurrentFrameMatchedInlier;
        std::timed_mutex mMutexCurrentFrameMapPointDepth;
        std::timed_mutex mMutexCurrentFrameFeatures;

        std::timed_mutex mMutexTotalNumKeyFrame;

        std::timed_mutex mMutexLocalMappingBANumber;
        std::timed_mutex mMutexLocalMappingBAOptimizer;

        std::timed_mutex mMutexLocalNewKFinQueueNumber;

        std::timed_mutex mMutexLoopClosureBAOptimizer;
        std::timed_mutex mMutexGlobalNewKFinQueueNumber;
        //std::timed_mutex mMutex;


        // binary flags for data collection status
        bool mbImageFeaturesReady;
        bool mbCurrentFrameFeaturesReady;
        bool mbIsNewFrameProcessed;


        // member variables for data collection
        // input features
        double mdTimeStamp;
        string msImgFileName;
        int mnImCounter;
        cv::Mat mImGrey;
        double mdBrightness;
        double mdContrast;
        double mdEntropy;
        double mdLaplacian;

        cv::Mat mImDistribution;

        // tracking features
        int mnTrackMode;
        int mnTrackMethod;
        int mnPrePOOutlier;
        int mnPrePOKeyMapLoss;
        int mnInlier;
        int mnPostPOOutlier;
        int mnPostPOKeyMapLoss;
        int mnMatchedInlier;

        Frame mCurrentFrame;
        int mnkeypoints;
        vector<float> mvMapPointMinDepth;
        float mfMapPointAvgMinDepth;
        float mfMapPointVarMinDepth;
        // Current camera pose in world reference
        Sophus::SE3f mTwc;
        // World Pose
        Eigen::Quaternionf mQ; //= Twc.unit_quaternion();
        Eigen::Vector3f mtwc; //= Twc.translation();
        // Relative Pose
        Eigen::Quaternionf mRQ;
        Eigen::Vector3f mRtwc;
        Eigen::Vector3f mREuler;

        KeyFrame mCurrentKeyFrame;

        int mnKeyFramesInMaps;

        // Local mapping features
        int mnFixedKF_BA;
        int mnOptKF_BA;
        int mnMPs_BA;
        int mnEdges_BA;
        int mnLocalBA;
        float mfLocalBAVisualError;
        int mnLocalNewKFinQueue;

        // Loop closure features
        int mnGlobalBA;
        float mfGlobalBAVisualError;
        int mnGlobalNewKFinQueue;


        // Private functions to process the collected data
        double CalculateImageEntropy(const cv::Mat &image);

        void CalculateImageFeatures();

        void CalculateCurrentFrameFeatures();

        float mfDuration;

        // Private member of the buffer
        //FrameBuffer mBuffer; // Create a buffer with a capacity of 30 frames


        // New Keyframe Generation Parameters
        float mfActionThRefRatio = 0.8;
        int mnActionMaxFrames = 30;
        int mnActionMinFrames = 0;

        // For TCP connection
        TCPClient* mpTCPClient;
        void InitializeTCPClient(const std::string& serverIP, int serverPort);
        void SendRowTCP();
        void SetTCP2Actions();
    };

}

#endif //ORB_SLAM3_DATACOLLECTING_H
