//
// Created by tianyi on 6/15/23.
//

#include "DataCollecting.h"

#include "Tracking.h"
#include "LocalMapping.h"
#include "LoopClosing.h"

#include "UdpSender.h"
#include "TCPClient.h"
#include "UdpPointDistSender.h"

namespace ORB_SLAM3
{

bool mbEnable = false;

DataCollecting::DataCollecting(System* pSys, Atlas *pAtlas, const float bMonocular, bool bInertial, const string &_strSeqName):
        mpSystem(pSys), mbMonocular(bMonocular), mbInertial(bInertial)
{
    if(mbEnable){
        mpAtlas = pAtlas;
        msSeqName = _strSeqName;

        InitializeDataCollector();
        InitializeCSVLogger();
        //InitializeUDPTSender();
        // InitializeTCPClient("127.0.0.1", 10000);
        // For Jetson --> Lambda connection
        // Use your server's IP address
        InitializeTCPClient("192.168.1.46", 10000);
        
        InitializeUDPPointDistSender();

        cout << "Data Collecting Module Initialized" << endl;
    }
    else{
        cout << "Data Collecting Module is NOT Initialized" << endl;
    }
}


void DataCollecting::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

void DataCollecting::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void DataCollecting::SetLoopCloser(LoopClosing *pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void DataCollecting::InitializeDataCollector()
{
    if(mbEnable){
        // binary flags for data collection status
        mbFinished = false;
        mbImageFeaturesReady = false;
        mbCurrentFrameFeaturesReady = false;
        mnImCounter = 0;
        mbIsNewFrameProcessed = true; // Initialized as it is already processed

        mnLocalBA = 0;

        // member variables for data collection
        // input features
        mdTimeStamp = 0.0;
        msImgFileName = "None";
        mnImCounter = 0;
        //cv::Mat mImGrey;
        mdBrightness = 0.0;
        mdContrast = 0.0;
        mdEntropy = 0.0;
        mdLaplacian = 0.0;

        // tracking features
        mnTrackMode = 0;
        mnTrackMethod = 0;
        mnPrePOOutlier = 0;
        mnPrePOKeyMapLoss = 0;
        mnInlier = 0;
        mnPostPOOutlier = 0;
        mnPostPOKeyMapLoss = 0;
        mnMatchedInlier = 0;

        mCurrentFrame.N = 0;
        mnkeypoints = 0;
        //vector<float> mvMapPointMinDepth;
        mfMapPointAvgMinDepth = 0.0;
        mfMapPointVarMinDepth = 0.0;
    }
    
}

void DataCollecting::Run()
{
    if(mbEnable){
        // Initialize the data collector
        cout << "Data Collecting Module Running" << endl;
        // wait for 1 second
        usleep(1*1000*1000);

        while(1)
        {
            if(!mbIsNewFrameProcessed)
            {
                auto start = chrono::high_resolution_clock::now();

                // If the new arrived frame haven't been processed
                if(mbImageFeaturesReady)
                {
                    CalculateImageFeatures();
                    /// For saving the point spatial distribution
                //    if(mImDistribution.empty() == false){
                //        std::ostringstream oss;
                //        oss << std::fixed << std::setprecision(6) << mdTimeStamp;
                //        std::string fileName = "./distribution/"+oss.str()+".png";
                //        cv::imwrite(fileName, mImDistribution);
                //    }
                    // Check if buffer is valid
                    // if (mBuffer.isValid()) {
                    //     std::ostringstream oss;
                    //     oss << std::fixed << std::setprecision(6) << mdTimeStamp;
                    //     std::string fileName = "./logs/distribution/"+oss.str()+".bin";
                    //     std::cout << "Buffer is full and valid. Saving specific frames to binary file..." << std::endl;
                    //     mBuffer.saveSnapshot(fileName);
                    // } else {
                    //     std::cout << "Buffer is not valid yet." << std::endl;
                    // }
                }
                if(mbCurrentFrameFeaturesReady)
                {
                    CalculateCurrentFrameFeatures();
                }
                if (mbImageFeaturesReady && mbCurrentFrameFeaturesReady)
                {
                    //WriteRowCSVLogger();
                    //SendRowUDP();
                    SendRowTCP();
                    if (!mImDistribution.empty()){
                        SendPointDistUDP(mImDistribution);
                    }
                }
                // New Case Study not using RL
                //SetTCP2Actions();

                // Try to acquire the timed_mutex with a timeout of 1 milliseconds
                if (mMutexNewFrameProcessed.try_lock_for(std::chrono::milliseconds(1))) {
                    mbIsNewFrameProcessed = true;
                    mMutexNewFrameProcessed.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexNewFrameProcessed within 1 milliseconds." << std::endl;
                }

                auto stop = chrono::high_resolution_clock::now();
                mfDuration = chrono::duration_cast<chrono::microseconds>(stop - start).count();
                //cout << "Run() elapsed: " << mfDuration << " microseconds" << endl;

            }
            usleep(0.005*1000*1000);
            // Wait for 0.005s = 5ms
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Define methods to collect features from ORB-SLAM3
//////////////////////////////////////////////////////////////////////////

void DataCollecting::CollectImageTimeStamp(const double &timestamp)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexImageTimeStamp.try_lock_for(std::chrono::milliseconds(1))) {
            mdTimeStamp = timestamp;
            mMutexImageTimeStamp.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexImageTimeStamp within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectImageFileName(string &filename)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexImageFileName.try_lock_for(std::chrono::milliseconds(1))) {
            msImgFileName = filename;
            mMutexImageFileName.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexImageFileName within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectImagePixel(cv::Mat &imGrey)
{
    if(mbEnable){
        // Try to acquire the mMutexImagePixel with a timeout of 1 milliseconds
        if (mMutexImagePixel.try_lock_for(std::chrono::milliseconds(1))) {
            //Option1: save the image with the same resolution
            mImGrey = imGrey.clone();

            //Option2: save the image with reduced resolution (reduced by 1/16 = 1/4*1/4)
            //cv::resize(imGrey, mImGrey, cv::Size(), 0.25, 0.25, cv::INTER_NEAREST);
            // Instead, reduce the image resolution later at the calculation step
            mMutexImagePixel.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexImagePixel within 1 milliseconds." << std::endl;
        }

        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexImageCounter.try_lock_for(std::chrono::milliseconds(1))) {
            mnImCounter ++;
            mMutexImageCounter.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexImageCounter within 1 milliseconds." << std::endl;
        }
        // update the data ready flag
        mbImageFeaturesReady = true;

        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexNewFrameProcessed.try_lock_for(std::chrono::milliseconds(1))) {
            mbIsNewFrameProcessed = false;
            mMutexNewFrameProcessed.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexNewFrameProcessed within 1 milliseconds." << std::endl;
        }

    }

}

void DataCollecting::CollectDistribution(cv::Mat &distMatrix)
{
    if(mbEnable){
        mImDistribution = distMatrix;
    //    // Display the updated matrix
    //    cv::namedWindow("Distribution", cv::WINDOW_AUTOSIZE);
    //    cv::Mat displayMatrix;
    //    int scale_factor = 10;
    //    cv::resize(mImDistribution, displayMatrix, cv::Size(FRAME_GRID_COLS * scale_factor, FRAME_GRID_ROWS * scale_factor),0, 0, cv::INTER_NEAREST);
    //    cv::imshow("Distribution", displayMatrix);
    //    
    //    mBuffer.insert(mImDistribution);
    }
}

void DataCollecting::CollectCurrentFrame(const Frame &frame)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFrame.try_lock_for(std::chrono::milliseconds(1))) {
            mCurrentFrame = frame;
            mbCurrentFrameFeaturesReady = true;
            mMutexCurrentFrame.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFrame within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFrameTrackMode(const int &nTrackMode)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFrameTrackMode.try_lock_for(std::chrono::milliseconds(1))) {
            mnTrackMode = nTrackMode;
            mMutexCurrentFrameTrackMode.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFrameTrackMode within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFrameTrackMethod(const int &nTrackMethod)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        // Track with motion model == 1
        // Track with reference keyframe == 2
        // Track with relocalization == 3
        if (mMutexCurrentFrameTrackMethod.try_lock_for(std::chrono::milliseconds(1))) {
            mnTrackMethod = nTrackMethod;
            mMutexCurrentFrameTrackMethod.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFrameTrackStep within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFramePrePOOutlier(const int &nPrePOOutlier)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFramePrePOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
            mnPrePOOutlier = nPrePOOutlier;
            mMutexCurrentFramePrePOOutlier.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFramePrePOOutlier within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFramePrePOKeyMapLoss(const int &nPrePOKeyMapLoss)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFramePrePOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
            mnPrePOKeyMapLoss = nPrePOKeyMapLoss;
            mMutexCurrentFramePrePOKeyMapLoss.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFramePrePOKeyMapLoss within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFrameInlier(const int &nInlier)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFrameInlier.try_lock_for(std::chrono::milliseconds(1))) {
            mnInlier = nInlier;
            mMutexCurrentFrameInlier.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFrameInlier within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFramePostPOOutlier(const int &nPostPOOutlier)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFramePostPOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
            mnPostPOOutlier = nPostPOOutlier;
            mMutexCurrentFramePostPOOutlier.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFramePostPOOutlier within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFramePostPOKeyMapLoss(const int &nPostPOKeyMapLoss)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFramePostPOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
            mnPostPOKeyMapLoss = nPostPOKeyMapLoss;
            mMutexCurrentFramePostPOKeyMapLoss.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFramePostPOKeyMapLoss within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFrameMatchedInlier(const int &nMatchedInlier)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFrameMatchedInlier.try_lock_for(std::chrono::milliseconds(1))) {
            mnMatchedInlier = nMatchedInlier;
            mMutexCurrentFrameMatchedInlier.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFrameMatchedInlier within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectCurrentFrameMapPointDepth(const Frame &frame)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFrameMapPointDepth.try_lock_for(std::chrono::milliseconds(1))) {
            vector<float> vMapPointMinDepth;
            for (int i = 0; i < frame.N; i++)
            {
                // If current frame i-th map point is not Null
                if (frame.mvpMapPoints[i])
                    vMapPointMinDepth.push_back(frame.mvpMapPoints[i]->GetMinDistanceInvariance());
            }
            // Calculate the mean
            mfMapPointAvgMinDepth = accumulate(vMapPointMinDepth.begin(), vMapPointMinDepth.end(), 0.0) / vMapPointMinDepth.size();
            // Calculate the variance
            float sum_of_squares = 0;
            for (auto num : vMapPointMinDepth)
            {
                sum_of_squares += pow(num - mfMapPointAvgMinDepth, 2);
            }
            mfMapPointVarMinDepth = sum_of_squares / vMapPointMinDepth.size();
            mMutexCurrentFrameMapPointDepth.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFrameMapPointDepth within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectLocalMappingBANumber(const int num_FixedKF_BA, const int num_OptKF_BA,
                                                 const int num_MPs_BA,  const int num_edges_BA)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexLocalMappingBANumber.try_lock_for(std::chrono::milliseconds(1))) {
            // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
            mnFixedKF_BA = num_FixedKF_BA;
            // Local keyframes
            mnOptKF_BA = num_OptKF_BA;
            mnMPs_BA = num_MPs_BA;
            mnEdges_BA = num_edges_BA;
            mMutexLocalMappingBANumber.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexLocalMappingBANumber within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectLocalMappingNewKFinQueue(const int num_NewKFinQueue)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexLocalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
            mnLocalNewKFinQueue = num_NewKFinQueue;
            mMutexLocalNewKFinQueueNumber.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexLocalNewKFinQueueNumber within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectLocalMappingBAOptimizer(const float fLocalBAVisualError)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexLocalMappingBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
            mfLocalBAVisualError = fLocalBAVisualError;
            mnLocalBA ++;
            mMutexLocalMappingBAOptimizer.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexLocalMappingBAOptimizer within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectLoopClosureBAOptimizer(const float fGlobalBAVisualError)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexLoopClosureBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
            mfGlobalBAVisualError = fGlobalBAVisualError;
            mnGlobalBA ++;
            mMutexLoopClosureBAOptimizer.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexLoopClosureBAOptimizer within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::CollectLoopClosureNewKFinQueue(const int num_NewKFinQueue)
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexGlobalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
            mnGlobalNewKFinQueue = num_NewKFinQueue;
            mMutexGlobalNewKFinQueueNumber.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexGlobalNewKFinQueueNumber within 1 milliseconds." << std::endl;
        }
    }
}
//////////////////////////////////////////////////////////////////////////
// END: Methods to collect features from ORB-SLAM3
//////////////////////////////////////////////////////////////////////////




//////////////////////////////////////////////////////////////////////////
// Methods to process features from ORB-SLAM3
//////////////////////////////////////////////////////////////////////////

double DataCollecting::CalculateImageEntropy(const cv::Mat& image)
{
    // Calculate histogram of pixel intensities
    cv::Mat hist;
    int histSize = 256;  // Number of bins for intensity values
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // Normalize histogram
    hist /= image.total();

    // Calculate entropy
    double entropy = 0;
    for (int i = 0; i < histSize; ++i)
    {
        if (hist.at<float>(i) > 0)
        {
            entropy -= hist.at<float>(i) * std::log2(hist.at<float>(i));
        }
    }
    return entropy;
}

void DataCollecting::CalculateImageFeatures()
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexImagePixel.try_lock_for(std::chrono::milliseconds(1))) {
            // Try to acquire the timed_mutex with a timeout of 1 milliseconds
            if (mMutexImageFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                if (mImGrey.empty())
                {
                    cout << "Failed to load image." << endl;
                }
                else {
                    cv::resize(mImGrey, mImGrey, cv::Size(), 0.2, 0.2, cv::INTER_NEAREST);
                    cv::Scalar meanValue, stddevValue;
                    // calculate the brightness and contrast
                    cv::meanStdDev(mImGrey, meanValue, stddevValue);
                    mdBrightness = meanValue[0];
                    mdContrast = stddevValue[0];
                    // calculate the entropy
                    mdEntropy = CalculateImageEntropy(mImGrey);

                    cv::Mat laplacianImage;
                    //cv::Laplacian(mImGrey, laplacianImage, CV_16S, 3);
                    // Since the radius of ORB feature extractor is 16?? Sorry must be odd number
                    cv::Laplacian(mImGrey, laplacianImage, CV_16S, 5);
                    cv::convertScaleAbs(laplacianImage, laplacianImage);
                    //cv::imshow("laplacian", laplacianImage);
                    //cv::waitKey(1);
                    cv::meanStdDev(laplacianImage, meanValue, stddevValue);
                    mdLaplacian = stddevValue[0]; //meanValue[0];
                }
                mMutexImageFeatures.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexImageFeatures within 1 milliseconds." << std::endl;
            }
            mMutexImagePixel.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexImagePixel within 1 milliseconds." << std::endl;
        }
    }
}


void DataCollecting::CalculateCurrentFrameFeatures()
{
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexCurrentFrame.try_lock_for(std::chrono::milliseconds(1))) {
            if(mbCurrentFrameFeaturesReady)
            {
                if(mnTrackMode==2){
                    if (mMutexCurrentFrameFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                        mnkeypoints = mCurrentFrame.N;
                        // current camera pose in world reference
                        Sophus::SE3f currentTwc = mCurrentFrame.GetPose().inverse();
                        // get current pose - Rotation
                        mQ = currentTwc.unit_quaternion();
                        //mtwc = currentTwc.translation();
                        // Get Relative Translation
                        // Project previous camera world position (mtwc) back to current camera coordinates
                        mRtwc = mCurrentFrame.inRefCoordinates(mtwc);
                        //mRtwc = currentTwc.translation() - mtwc;
                        // update current pose - Translation
                        mtwc = currentTwc.translation();

                        // calculate relative pose
                        Sophus::SE3f currentRelativePose = mTwc.inverse() * currentTwc;
                        Eigen::Matrix3f rotationMatrix = currentRelativePose.rotationMatrix();
                        mREuler = rotationMatrix.eulerAngles(2, 1, 0); // ZYX convention
                        
                        // update current camera pose
                        mTwc = currentTwc;

                        mMutexCurrentFrameFeatures.unlock();
                    }
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameFeatures within 1 milliseconds." << std::endl;
                }
            }
            mMutexCurrentFrame.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexCurrentFrame within 1 milliseconds." << std::endl;
        }
    }
}

void DataCollecting::RequestFinish()
{
    mFileLogger.close();
    mbFinished = true;

}

//////////////////////////////////////////////////////////////////////////
// START: Methods to save features in .CSV
//////////////////////////////////////////////////////////////////////////

void DataCollecting::CollectCurrentTime()
{
    if(mbEnable){
        // Get the current time
        std::time_t currentTime = std::time(nullptr);

        // Convert the time to a string
        std::tm* now = std::localtime(&currentTime);

        // Format the date and time string
        std::ostringstream oss;
        oss << std::put_time(now, "%y%m%d_%H%M%S");
        msCurrentTime = oss.str();
    }
}

void DataCollecting::InitializeCSVLogger()
{
    if(mbEnable){
        CollectCurrentTime();
        //msCSVFileName = msSeqName + "_" + msCurrentTime + ".csv";
        msCSVFileName = "./logs/log.csv";

        // Open the CSV file for writing
        mFileLogger.open(msCSVFileName);

        // Write the first row with column names
        if (!mFileLogger.is_open())
        {
            std::cerr << "Unable to open file: " << msCSVFileName << std::endl;
        }
        else
        {
            // Write column names
    //        for (const auto& columnName : mvsColumnFeatureNames)
    //        {
    //            mFileLogger << columnName << ",";
    //        }
    //        mFileLogger << endl;
            size_t totalSize = mvsColumnFeatureNames.size();
            for (size_t i = 0; i < totalSize; ++i) {
                const auto& columnName = mvsColumnFeatureNames[i];
                if (i < totalSize-1){
                    mFileLogger << columnName << ",";
                }
                else{
                    mFileLogger << columnName;
                }
            }
            mFileLogger << endl;
        }
    }
}


void DataCollecting::InitializeUDPTSender() {
    if(mbEnable){
        // Create a socket
        mpUdpSender = new UdpSender("127.0.0.1", 6660);
    }
}

void DataCollecting::InitializeUDPPointDistSender() {
    if(mbEnable){
        // Create a socket
        // mpUdpPointDistSender = new UdpPointDistSender("127.0.0.1", 65535);
        // For Jetson to Lambda
        mpUdpPointDistSender = new UdpPointDistSender("192.168.1.46", 65535);
    }
}

void DataCollecting::InitializeTCPClient(const std::string& serverIP, int serverPort) {
    if(mbEnable){
        // Create a socket
        mpTCPClient = new TCPClient(serverIP, serverPort);
        mpTCPClient->Connect();
    }
}


void DataCollecting::WriteRowCSVLogger()
{
    if(mbEnable){
        if (!mFileLogger.is_open())
        {
            std::cerr << "Unable to open file: " << msCSVFileName << std::endl;
        }
        else
        {
            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexImageCounter.try_lock_for(std::chrono::milliseconds(1))) {
                mFileLogger << mnImCounter << ",";
                mMutexImageCounter.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexImageCounter in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexImageTimeStamp.try_lock_for(std::chrono::milliseconds(1))) {
                mFileLogger << fixed << setprecision(6) << mdTimeStamp << ","; //1e9*mdTimeStamp << ",";
                mMutexImageTimeStamp.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexImageTimeStamp in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexCurrentFrameTrackMode.try_lock_for(std::chrono::milliseconds(1))) {
                mFileLogger << mnTrackMode << ",";
                mMutexCurrentFrameTrackMode.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexCurrentFrameTrackMode in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexCurrentFrameTrackMethod.try_lock_for(std::chrono::milliseconds(1))) {
                mFileLogger << mnTrackMethod;
                mMutexCurrentFrameTrackMethod.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexCurrentFrameTrackMethod in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            if(mbImageFeaturesReady)
            {
                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexImageFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << "," << fixed << std::setprecision(6);
                    mFileLogger << mdBrightness << "," << mdContrast << "," << mdEntropy << "," << mdLaplacian;

                    mMutexImageFeatures.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexImageFeatures in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }
            }

            if(mbCurrentFrameFeaturesReady)
            {
                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameMapPointDepth.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger  << "," << mfMapPointAvgMinDepth << "," << mfMapPointVarMinDepth << ",";
                    mMutexCurrentFrameMapPointDepth.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameMapPointDepth in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePrePOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnPrePOOutlier << ",";
                    mMutexCurrentFramePrePOOutlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePrePOOutlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePrePOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnPrePOKeyMapLoss << ",";
                    mMutexCurrentFramePrePOKeyMapLoss.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePrePOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameInlier.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnInlier << ",";
                    mMutexCurrentFrameInlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePrePOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePostPOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnPostPOOutlier << ",";
                    mMutexCurrentFramePostPOOutlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePostPOOutlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePostPOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnPostPOKeyMapLoss << ",";
                    mMutexCurrentFramePostPOKeyMapLoss.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePostPOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameMatchedInlier.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnMatchedInlier << ",";
                    mMutexCurrentFrameMatchedInlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameMatchedInlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnkeypoints << ",";
                    mFileLogger << setprecision(9) << mtwc(0) << "," << mtwc(1) << "," << mtwc(2) << ",";
                    mFileLogger << mQ.x() << "," << mQ.y() << "," << mQ.z() << "," << mQ.w() << ",";
                    mFileLogger << mRtwc(0) << "," << mRtwc(1) << "," << mRtwc(2) << ",";
                    mFileLogger << mREuler(0) << "," << mREuler(1) << "," << mREuler(2) << ",";
                    mMutexCurrentFrameFeatures.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameFeatures in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 1 milliseconds
                if (mMutexLocalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnLocalNewKFinQueue << ",";
                    mMutexLocalNewKFinQueueNumber.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLocalNewKFinQueueNumber within 1 milliseconds." << std::endl;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexLocalMappingBANumber.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnFixedKF_BA << "," << mnOptKF_BA << "," << mnMPs_BA << "," << mnEdges_BA << ",";
                    mMutexLocalMappingBANumber.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLocalMappingBANumber in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexLocalMappingBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnLocalBA << "," << mfLocalBAVisualError << ",";
                    mMutexLocalMappingBAOptimizer.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLocalMappingBAOptimizer in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 1 milliseconds
                if (mMutexGlobalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnGlobalNewKFinQueue << ",";
                    mMutexGlobalNewKFinQueueNumber.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexGlobalNewKFinQueueNumber within 1 milliseconds." << std::endl;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexLoopClosureBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
                    mFileLogger << mnGlobalBA << "," << mfGlobalBAVisualError << ",";
                    mMutexLoopClosureBAOptimizer.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLoopClosureBAOptimizer in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                mFileLogger << mfDuration << ",";

                mFileLogger << mfActionThRefRatio << "," << mnActionMinFrames << "," << mnActionMaxFrames;

            }

            mFileLogger << endl;
        }
    }
}



void DataCollecting::SendRowUDP()
{
    if(mbEnable){
        if (!mpUdpSender)
        {
            std::cerr << "UDP port haven't initialized" << std::endl;
        }
        else
        {
            std::ostringstream csvRow;

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexImageCounter.try_lock_for(std::chrono::milliseconds(1))) {
                csvRow << mnImCounter << ",";
                mMutexImageCounter.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexImageCounter in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexImageTimeStamp.try_lock_for(std::chrono::milliseconds(1))) {
                csvRow << fixed << setprecision(6) << mdTimeStamp << ","; //1e9*mdTimeStamp << ",";
                mMutexImageTimeStamp.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexImageTimeStamp in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexCurrentFrameTrackMode.try_lock_for(std::chrono::milliseconds(1))) {
                csvRow << mnTrackMode << ",";
                mMutexCurrentFrameTrackMode.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexCurrentFrameTrackMode in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexCurrentFrameTrackMethod.try_lock_for(std::chrono::milliseconds(1))) {
                csvRow << mnTrackMethod;
                mMutexCurrentFrameTrackMethod.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexCurrentFrameTrackMethod in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            if(mbImageFeaturesReady)
            {
                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexImageFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << "," << fixed << std::setprecision(6);
                    csvRow << mdBrightness << "," << mdContrast << "," << mdEntropy << "," << mdLaplacian;

                    mMutexImageFeatures.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexImageFeatures in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }
            }

            if(mbCurrentFrameFeaturesReady)
            {
                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameMapPointDepth.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow  << "," << mfMapPointAvgMinDepth << "," << mfMapPointVarMinDepth << ",";
                    mMutexCurrentFrameMapPointDepth.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameMapPointDepth in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePrePOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnPrePOOutlier << ",";
                    mMutexCurrentFramePrePOOutlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePrePOOutlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePrePOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnPrePOKeyMapLoss << ",";
                    mMutexCurrentFramePrePOKeyMapLoss.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePrePOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameInlier.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnInlier << ",";
                    mMutexCurrentFrameInlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePrePOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePostPOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnPostPOOutlier << ",";
                    mMutexCurrentFramePostPOOutlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePostPOOutlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePostPOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnPostPOKeyMapLoss << ",";
                    mMutexCurrentFramePostPOKeyMapLoss.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePostPOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameMatchedInlier.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnMatchedInlier << ",";
                    mMutexCurrentFrameMatchedInlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameMatchedInlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnkeypoints << ",";
                    csvRow << setprecision(9) << mtwc(0) << "," << mtwc(1) << "," << mtwc(2) << ",";
                    csvRow << mQ.x() << "," << mQ.y() << "," << mQ.z() << "," << mQ.w() << ",";
                    csvRow << mRtwc(0) << "," << mRtwc(1) << "," << mRtwc(2) << ",";
                    csvRow << mREuler(0) << "," << mREuler(1) << "," << mREuler(2) << ",";
                    mMutexCurrentFrameFeatures.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameFeatures in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 1 milliseconds
                if (mMutexLocalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnLocalNewKFinQueue << ",";
                    mMutexLocalNewKFinQueueNumber.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLocalNewKFinQueueNumber within 1 milliseconds." << std::endl;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexLocalMappingBANumber.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnFixedKF_BA << "," << mnOptKF_BA << "," << mnMPs_BA << "," << mnEdges_BA << ",";
                    mMutexLocalMappingBANumber.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLocalMappingBANumber in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexLocalMappingBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnLocalBA << "," << mfLocalBAVisualError << ",";
                    mMutexLocalMappingBAOptimizer.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLocalMappingBAOptimizer in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 1 milliseconds
                if (mMutexGlobalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnGlobalNewKFinQueue << ",";
                    mMutexGlobalNewKFinQueueNumber.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexGlobalNewKFinQueueNumber within 1 milliseconds." << std::endl;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexLoopClosureBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnGlobalBA << "," << mfGlobalBAVisualError << ",";
                    mMutexLoopClosureBAOptimizer.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLoopClosureBAOptimizer in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                csvRow << mfDuration;

            }

            std::string csvString = csvRow.str();
            mpUdpSender->sendCSVRow(csvString);
//            ssize_t sentBytes = sendto(sockfd, csvString.c_str(), csvString.size(), 0, (struct sockaddr*)&destAddr, sizeof(destAddr));
//            if (sentBytes < 0) {
//                std::cerr << "Error sending data." << std::endl;
//            }
        }
    }
}


void DataCollecting::SendRowTCP()
{
    if(mbEnable){
        if (!mpTCPClient)
        {
            std::cerr << "TCP port haven't initialized" << std::endl;
        }
        else
        {
            std::ostringstream csvRow;

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexImageCounter.try_lock_for(std::chrono::milliseconds(1))) {
                csvRow << mnImCounter << ",";
                mMutexImageCounter.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexImageCounter in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexImageTimeStamp.try_lock_for(std::chrono::milliseconds(1))) {
                csvRow << fixed << setprecision(6) << mdTimeStamp << ","; //1e9*mdTimeStamp << ",";
                mMutexImageTimeStamp.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexImageTimeStamp in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
            if (mMutexCurrentFrameTrackMode.try_lock_for(std::chrono::milliseconds(1))) {
                csvRow << mnTrackMode << ",";
                mMutexCurrentFrameTrackMode.unlock();
            }
            else {
                std::cout << " failed to acquire the mMutexCurrentFrameTrackMode in CSV logger within 1 milliseconds." << std::endl;
                return;
            }

            if(mbImageFeaturesReady)
            {
                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexImageFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << fixed << std::setprecision(6);
                    csvRow << mdBrightness << "," << mdContrast << "," << mdEntropy << "," << mdLaplacian;

                    mMutexImageFeatures.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexImageFeatures in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }
            }

            if(mbCurrentFrameFeaturesReady)
            {
                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameMapPointDepth.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow  << "," << mfMapPointAvgMinDepth << "," << mfMapPointVarMinDepth << ",";
                    mMutexCurrentFrameMapPointDepth.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameMapPointDepth in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

    //            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
    //            if (mMutexCurrentFramePrePOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
    //                csvRow << mnPrePOOutlier << ",";
    //                mMutexCurrentFramePrePOOutlier.unlock();
    //            }
    //            else {
    //                std::cout << " failed to acquire the mMutexCurrentFramePrePOOutlier in CSV logger within 1 milliseconds." << std::endl;
    //                return;
    //            }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePrePOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnPrePOKeyMapLoss << ",";
                    mMutexCurrentFramePrePOKeyMapLoss.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePrePOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

    //            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
    //            if (mMutexCurrentFrameInlier.try_lock_for(std::chrono::milliseconds(1))) {
    //                csvRow << mnInlier << ",";
    //                mMutexCurrentFrameInlier.unlock();
    //            }
    //            else {
    //                std::cout << " failed to acquire the mMutexCurrentFramePrePOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
    //                return;
    //            }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFramePostPOOutlier.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnPostPOOutlier << ",";
                    mMutexCurrentFramePostPOOutlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFramePostPOOutlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

    //            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
    //            if (mMutexCurrentFramePostPOKeyMapLoss.try_lock_for(std::chrono::milliseconds(1))) {
    //                csvRow << mnPostPOKeyMapLoss << ",";
    //                mMutexCurrentFramePostPOKeyMapLoss.unlock();
    //            }
    //            else {
    //                std::cout << " failed to acquire the mMutexCurrentFramePostPOKeyMapLoss in CSV logger within 1 milliseconds." << std::endl;
    //                return;
    //            }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameMatchedInlier.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << mnMatchedInlier << ",";
                    mMutexCurrentFrameMatchedInlier.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameMatchedInlier in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexCurrentFrameFeatures.try_lock_for(std::chrono::milliseconds(1))) {
                    //csvRow << mnkeypoints << ",";
                    //csvRow << setprecision(9) << mtwc(0) << "," << mtwc(1) << "," << mtwc(2) << ",";
                    //csvRow << mQ.x() << "," << mQ.y() << "," << mQ.z() << "," << mQ.w() << ",";
                    csvRow << mRtwc(0) << "," << mRtwc(1) << "," << mRtwc(2) << ",";
                    csvRow << mREuler(0) << "," << mREuler(1) << "," << mREuler(2) << ",";
                    mMutexCurrentFrameFeatures.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexCurrentFrameFeatures in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

    //            // Try to acquire the timed_mutex with a timeout of 1 milliseconds
    //            if (mMutexLocalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
    //                csvRow << mnLocalNewKFinQueue << ",";
    //                mMutexLocalNewKFinQueueNumber.unlock();
    //            }
    //            else {
    //                std::cout << " failed to acquire the mMutexLocalNewKFinQueueNumber within 1 milliseconds." << std::endl;
    //            }

    //            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
    //            if (mMutexLocalMappingBANumber.try_lock_for(std::chrono::milliseconds(1))) {
    //                csvRow << mnFixedKF_BA << "," << mnOptKF_BA << "," << mnMPs_BA << "," << mnEdges_BA << ",";
    //                mMutexLocalMappingBANumber.unlock();
    //            }
    //            else {
    //                std::cout << " failed to acquire the mMutexLocalMappingBANumber in CSV logger within 1 milliseconds." << std::endl;
    //                return;
    //            }

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexLocalMappingBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
                    //csvRow << mnLocalBA << ",";
                    csvRow << mfLocalBAVisualError << ",";
                    mMutexLocalMappingBAOptimizer.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLocalMappingBAOptimizer in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

    //            // Try to acquire the timed_mutex with a timeout of 1 milliseconds
    //            if (mMutexGlobalNewKFinQueueNumber.try_lock_for(std::chrono::milliseconds(1))) {
    //                csvRow << mnGlobalNewKFinQueue << ",";
    //                mMutexGlobalNewKFinQueueNumber.unlock();
    //            }
    //            else {
    //                std::cout << " failed to acquire the mMutexGlobalNewKFinQueueNumber within 1 milliseconds." << std::endl;
    //            }

    //            // Try to acquire the timed_mutex with a timeout of 10 milliseconds
    //            if (mMutexLoopClosureBAOptimizer.try_lock_for(std::chrono::milliseconds(1))) {
    //                csvRow << mnGlobalBA << "," << mfGlobalBAVisualError << ",";
    //                mMutexLoopClosureBAOptimizer.unlock();
    //            }
    //            else {
    //                std::cout << " failed to acquire the mMutexLoopClosureBAOptimizer in CSV logger within 1 milliseconds." << std::endl;
    //                return;
    //            }

    //            csvRow << mfDuration << ",";

                csvRow << mfActionThRefRatio << "," << mnActionMinFrames << "," << mnActionMaxFrames;

                // Try to acquire the timed_mutex with a timeout of 10 milliseconds
                if (mMutexTotalNumKeyFrame.try_lock_for(std::chrono::milliseconds(1))) {
                    csvRow << ",";
                    csvRow << mnKeyFramesInMaps;
                    mMutexTotalNumKeyFrame.unlock();
                }
                else {
                    std::cout << " failed to acquire the mMutexLoopClosureBAOptimizer in CSV logger within 1 milliseconds." << std::endl;
                    return;
                }

            }

            std::string csvString = csvRow.str();
            mpTCPClient->SendMessage(csvString);
        }
    }
}


void DataCollecting::SetTCP2Actions()
{
    if(mbEnable){
        std::string receivedMessage;
        if (mpTCPClient->ReceiveMessage(receivedMessage)) {
            std::vector<float> floats;

            std::istringstream ss(receivedMessage);
            std::string token;

            while (std::getline(ss, token, ',')) {
                try {
                    float value = std::stof(token);
                    floats.push_back(value);
                } catch (const std::invalid_argument& e) {
                    // Handle invalid float value
                    std::cerr << "Invalid float value: " << token << std::endl;
                }
            }
    //        std::cout << "Receive actions: ";
    //        for (float f : floats) {
    //            std::cout << setprecision(3) << f << " ";
    //        }
    //
    //        std::cout << std::endl;

            mfActionThRefRatio = floats[0];
            mnActionMinFrames = (int)floats[1];
            mnActionMaxFrames = (int)floats[2];
        }
    }
}


void DataCollecting::CollectTotalNumKeyFrames(int keyFrameinMap){
    if(mbEnable){
        // Try to acquire the timed_mutex with a timeout of 1 milliseconds
        if (mMutexTotalNumKeyFrame.try_lock_for(std::chrono::milliseconds(1))) {
            mnKeyFramesInMaps = keyFrameinMap;
            mMutexTotalNumKeyFrame.unlock();
        }
        else {
            std::cout << " failed to acquire the mMutexTotalNumKeyFrame within 1 milliseconds." << std::endl;
        }
    }
}



void DataCollecting::SendPointDistUDP(const cv::Mat& pointDist){
    if(mbEnable){
        if (!mpUdpPointDistSender->sendPointDist(pointDist, ".png")) {
            std::cerr << "Failed to send pointDist." << std::endl;
        }
    }
}


float DataCollecting::GetActionThRefRatio()
{
    return mfActionThRefRatio;
}

int DataCollecting::GetActionMaxFrames()
{
    return mnActionMaxFrames;
}

int DataCollecting::GetActionMinFrames()
{
    return mnActionMinFrames;
}


// End of the ORBSLAM3 namespace
}
