/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "IOWrapper/ImageDisplay.h"
#include "stdio.h"
#include "util/globalCalib.h"
#include "util/globalFuncs.h"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/CoarseTracker.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include "util/DatasetReader.h"

#include <cmath>
#include <cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

namespace dso {
    int FrameHessian::instanceCounter = 0;
    int PointHessian::instanceCounter = 0;
    int CalibHessian::instanceCounter = 0;

    FullSystem::FullSystem() {

        int retstat = 0;
        if (setting_logStuff) {

            retstat += system("rm -rf logs");
            retstat += system("mkdir logs");

            retstat += system("rm -rf mats");
            retstat += system("mkdir mats");

            calibLog = new std::ofstream();
            calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
            calibLog->precision(12);

            numsLog = new std::ofstream();
            numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
            numsLog->precision(10);

            coarseTrackingLog = new std::ofstream();
            coarseTrackingLog->open("logs/coarseTrackingLog.txt",
                                    std::ios::trunc | std::ios::out);
            coarseTrackingLog->precision(10);

            eigenAllLog = new std::ofstream();
            eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
            eigenAllLog->precision(10);

            eigenPLog = new std::ofstream();
            eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
            eigenPLog->precision(10);

            eigenALog = new std::ofstream();
            eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
            eigenALog->precision(10);

            DiagonalLog = new std::ofstream();
            DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
            DiagonalLog->precision(10);

            variancesLog = new std::ofstream();
            variancesLog->open("logs/variancesLog.txt",
                               std::ios::trunc | std::ios::out);
            variancesLog->precision(10);

            nullspacesLog = new std::ofstream();
            nullspacesLog->open("logs/nullspacesLog.txt",
                                std::ios::trunc | std::ios::out);
            nullspacesLog->precision(10);
        } else {
            nullspacesLog = 0;
            variancesLog = 0;
            DiagonalLog = 0;
            eigenALog = 0;
            eigenPLog = 0;
            eigenAllLog = 0;
            numsLog = 0;
            calibLog = 0;
        }

        assert(retstat != 293847);

        selectionMap = new float[wG[0] * hG[0]];

        coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
        coarseTracker = new CoarseTracker(wG[0], hG[0]);
        coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
        coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
        pixelSelector = new PixelSelector(wG[0], hG[0]);

        statistics_lastNumOptIts = 0;
        statistics_numDroppedPoints = 0;
        statistics_numActivatedPoints = 0;
        statistics_numCreatedPoints = 0;
        statistics_numForceDroppedResBwd = 0;
        statistics_numForceDroppedResFwd = 0;
        statistics_numMargResFwd = 0;
        statistics_numMargResBwd = 0;

        lastCoarseRMSE.setConstant(100);

        currentMinActDist = 2;
        initialized = false;

        ef = new EnergyFunctional();
        ef->red = &this->treadReduce;

        isLost = false;
        initFailed = false;

        needNewKFAfter = -1;

        linearizeOperation = true;
        runMapping = true;
        mappingThread = boost::thread(&FullSystem::mappingLoop, this);
        lastRefStopID = 0;

        minIdJetVisDebug = -1;
        maxIdJetVisDebug = -1;
        minIdJetVisTracker = -1;
        maxIdJetVisTracker = -1;
    }

    FullSystem::~FullSystem() {
        blockUntilMappingIsFinished();

        if (setting_logStuff) {
            calibLog->close();
            delete calibLog;
            numsLog->close();
            delete numsLog;
            coarseTrackingLog->close();
            delete coarseTrackingLog;
            // errorsLog->close(); delete errorsLog;
            eigenAllLog->close();
            delete eigenAllLog;
            eigenPLog->close();
            delete eigenPLog;
            eigenALog->close();
            delete eigenALog;
            DiagonalLog->close();
            delete DiagonalLog;
            variancesLog->close();
            delete variancesLog;
            nullspacesLog->close();
            delete nullspacesLog;
        }

        delete[] selectionMap;

        for (FrameShell *s : allFrameHistory)
            delete s;
        for (FrameHessian *fh : unmappedTrackedFrames)
            delete fh;

        delete coarseDistanceMap;
        delete coarseTracker;
        delete coarseTracker_forNewKF;
        delete coarseInitializer;
        delete pixelSelector;
        delete ef;
    }

    void FullSystem::setOriginalCalib(VecXf originalCalib, int originalW,
                                      int originalH) {}

    void FullSystem::setGammaFunction(float *BInv) {
        if (BInv == 0)
            return;

        // copy BInv.
        memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

        // invert.
        for (int i = 1; i < 255; i++) {
            // find val, such that Binv[val] = i.
            // I dont care about speed for this, so do it the stupid way.

            for (int s = 1; s < 255; s++) {
                if (BInv[s] <= i && BInv[s + 1] >= i) {
                    Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                    break;
                }
            }
        }
        Hcalib.B[0] = 0;
        Hcalib.B[255] = 255;
    }

    void FullSystem::printResult(std::string file) {
        boost::unique_lock<boost::mutex> lock(trackMutex);
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

        std::ofstream myfile;
        myfile.open(file.c_str());
        myfile << std::setprecision(15);
        int i = 0;

        Eigen::Matrix<double, 3, 3> last_R =
                (*(allFrameHistory.begin()))->camToWorld.so3().matrix();
        Eigen::Matrix<double, 3, 1> last_T =
                (*(allFrameHistory.begin()))->camToWorld.translation().transpose();

        for (FrameShell *s : allFrameHistory) {
            if (!s->poseValid) {
                myfile << last_R(0, 0) << " " << last_R(0, 1) << " " << last_R(0, 2)
                       << " " << last_T(0, 0) << " " << last_R(1, 0) << " "
                       << last_R(1, 1) << " " << last_R(1, 2) << " " << last_T(1, 0)
                       << " " << last_R(2, 0) << " " << last_R(2, 1) << " "
                       << last_R(2, 2) << " " << last_T(2, 0) << "\n";
                continue;
            }

            if (setting_onlyLogKFPoses && s->marginalizedAt == s->id) {
                myfile << last_R(0, 0) << " " << last_R(0, 1) << " " << last_R(0, 2)
                       << " " << last_T(0, 0) << " " << last_R(1, 0) << " "
                       << last_R(1, 1) << " " << last_R(1, 2) << " " << last_T(1, 0)
                       << " " << last_R(2, 0) << " " << last_R(2, 1) << " "
                       << last_R(2, 2) << " " << last_T(2, 0) << "\n";
                continue;
            }

            const Eigen::Matrix<double, 3, 3> R = s->camToWorld.so3().matrix();
            const Eigen::Matrix<double, 3, 1> T =
                    s->camToWorld.translation().transpose();

            last_R = R;
            last_T = T;

            myfile << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << T(0, 0)
                   << " " << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " "
                   << T(1, 0) << " " << R(2, 0) << " " << R(2, 1) << " " << R(2, 2)
                   << " " << T(2, 0) << "\n";

            //		myfile << s->timestamp <<
            //			" " << s->camToWorld.translation().transpose()<<
            //			" " << s->camToWorld.so3().unit_quaternion().x()<<
            //			" " << s->camToWorld.so3().unit_quaternion().y()<<
            //			" " << s->camToWorld.so3().unit_quaternion().z()<<
            //			" " << s->camToWorld.so3().unit_quaternion().w() <<
            //"\n";
            i++;
        }
        myfile.close();
    }

    Vec4 FullSystem::trackNewCoarse(FrameHessian *fh, FrameHessian *fh_right) {

        assert(allFrameHistory.size() > 0);
        // set pose initialization.

        for (IOWrap::Output3DWrapper *ow : outputWrapper) {
            ow->pushStereoLiveFrame(fh, fh_right);
            // ow->pushLiveFrame(fh);
        }

        int i = 0;

        FrameHessian *lastF = coarseTracker->lastRef;

        AffLight aff_last_2_l = AffLight(0, 0);

        std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

        if (allFrameHistory.size() == 2) {
            initializeFromInitializer(fh);

            lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(),
                                           Eigen::Matrix<double, 3, 1>::Zero()));

            for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02) {
                lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
            }

            coarseTracker->makeK(&Hcalib);
            coarseTracker->setCTRefForFirstFrame(frameHessians);

            lastF = coarseTracker->lastRef;
        } else {
            FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
            FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];
            SE3 slast_2_sprelast;
            SE3 lastF_2_slast;
            { // lock on global pose consistency!
                boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
                lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
                aff_last_2_l = slast->aff_g2l;
            }
            //先假设fh_2_slat是slast_2_sprelast
            SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

            //===================== test ========================

            //        int fh_id = fh->shell->id;
            //
            //        std::string gtPath = "/home/jiatianwu/dso/05/05.txt";
            //        std::ifstream ReadFile(gtPath.c_str());
            //        std::string temp;
            //        std::string delim (" ");
            //
            //        int counter = 0;
            //        while(std::getline(ReadFile, temp) && counter < fh_id )
            //        {
            //            counter++;
            //        }
            //        ReadFile.close();
            //
            //        std::vector<std::string> results;
            //        split(temp, delim, results);
            //
            //        double gtX = atof(results[3].c_str());
            //        double gtY = atof(results[7].c_str());
            //        double gtZ = atof(results[11].c_str());
            //        Eigen::Vector3d gtT(gtX, gtY, gtZ);
            //        Eigen::Matrix<double, 3, 3> gtR;
            //        for (int i = 0; i < 3; i++)
            //            for(int j = 0; j < 3; j++)
            //            {
            //                gtR(i, j) = atof(results[4*i + j].c_str());
            //            }
            //        SE3 gtPose (gtR, gtT);
            //
            //		std::cout << " is empty " <<
            // fh->shell->trackingRef->camToWorld.inverse() << std::endl;
            //		if(fh_id > 10)
            //		{
            //			lastF_2_fh_tries.push_back(fh->shell->trackingRef->camToWorld.inverse()
            //* gtPose);
            //		}

            // get last delta-movement.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() *
                                       lastF_2_slast); // assume constant motion.
            lastF_2_fh_tries.push_back(
                    fh_2_slast.inverse() * fh_2_slast.inverse() *
                    lastF_2_slast); // assume double motion (frame skipped)
            lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() *
                                       lastF_2_slast); // assume half motion.
            lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
            lastF_2_fh_tries.push_back(SE3());         // assume zero motion FROM KF.

            /*        lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse()
               * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() *
               fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
                    lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse()
               * SE3::exp(fh_2_slast.log()*1.5).inverse() *
               SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() *
               fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() *
               lastF_2_slast);
                    lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse()
               * SE3::exp(fh_2_slast.log()*1.5).inverse() *
               SE3::exp(fh_2_slast.log()*1.5).inverse() *
               SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

                    lastF_2_fh_tries.push_back(fh_2_slast.inverse() *
               fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() *
               fh_2_slast.inverse() * lastF_2_slast);
                    lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse()
               * SE3::exp(fh_2_slast.log()*1.5).inverse() *
               SE3::exp(fh_2_slast.log()*1.5).inverse() *
               SE3::exp(fh_2_slast.log()*1.5).inverse() *
               SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);*/

            // just try a TON of different initializations (all rotations). In the end,
            // if they don't work they will only be tried on the coarsest level, which
            // is super fast anyway. also, if tracking rails here we loose, so we
            // really, really want to avoid that.
            for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++) {
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, rotDelta, 0, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, rotDelta, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, rotDelta),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast *
                                           SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta),
                                               Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
                lastF_2_fh_tries.push_back(
                        fh_2_slast.inverse() * lastF_2_slast *
                        SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta),
                            Vec3(0, 0, 0))); // assume constant motion.
            }

            if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
                lastF_2_fh_tries.clear();
                lastF_2_fh_tries.push_back(SE3());
            }
        }

        Vec3 flowVecs = Vec3(100, 100, 100);
        SE3 lastF_2_fh = SE3();
        AffLight aff_g2l = AffLight(0, 0);

        // as long as maxResForImmediateAccept is not reached, I'll continue through
        // the options. I'll keep track of the so-far best achieved residual for each
        // level in achievedRes. If on a coarse level, tracking is WORSE than
        // achievedRes, we will not continue to save time.

        Vec5 achievedRes = Vec5::Constant(NAN);
        bool haveOneGood = false;
        int tryIterations = 0;
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++) {
            AffLight aff_g2l_this = aff_last_2_l;
            SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

            //在这里判断估计的RT矩阵是否正确
            bool trackingIsGood = coarseTracker->trackNewestCoarse(
                    fh, lastF_2_fh_this, aff_g2l_this, pyrLevelsUsed - 1,
                    achievedRes); // in each level has to be at least as good as the last
            // try.
            tryIterations++;

            if (i != 0) {
                printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f "
                               "%f): %f %f %f %f %f -> %f %f %f %f %f \n",
                       i, i, pyrLevelsUsed - 1, aff_g2l_this.a, aff_g2l_this.b,
                       achievedRes[0], achievedRes[1], achievedRes[2], achievedRes[3],
                       achievedRes[4], coarseTracker->lastResiduals[0],
                       coarseTracker->lastResiduals[1], coarseTracker->lastResiduals[2],
                       coarseTracker->lastResiduals[3], coarseTracker->lastResiduals[4]);
            }

            // do we have a new winner?判断估计的RT矩阵是否比目前最好的还要好
            if (trackingIsGood &&
                std::isfinite((float) coarseTracker->lastResiduals[0]) &&
                !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
                // printf("take over. minRes %f -> %f!\n", achievedRes[0],
                // coarseTracker->lastResiduals[0]);
                flowVecs = coarseTracker->lastFlowIndicators;
                aff_g2l = aff_g2l_this;
                lastF_2_fh = lastF_2_fh_this;
                haveOneGood = true;
            }

            // take over achieved res (always). 是的话替换achievedRes
            if (haveOneGood) {
                for (int i = 0; i < 5; i++) {
                    if (!std::isfinite((float) achievedRes[i]) ||
                        achievedRes[i] > coarseTracker->lastResiduals[i]) // take over if
                        // achievedRes is
                        // either bigger
                        // or NAN.
                        achievedRes[i] = coarseTracker->lastResiduals[i];
                }
            }

            if (haveOneGood &&
                achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
                break;
        }

        if (!haveOneGood) {
            printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope "
                           "we may somehow recover.\n");
            flowVecs = Vec3(0, 0, 0);
            aff_g2l = aff_last_2_l;
            lastF_2_fh = lastF_2_fh_tries[0];
        }

        lastCoarseRMSE = achievedRes;

        // no lock required, as fh is not used anywhere yet.
        fh->shell->camToTrackingRef = lastF_2_fh.inverse();
        fh->shell->trackingRef = lastF->shell;
        fh->shell->aff_g2l = aff_g2l;
        fh->shell->camToWorld =
                fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

        Eigen::Matrix<double, 3, 1> last_T =
                fh->shell->camToWorld.translation().transpose();
        std::cout << "x:" << last_T(0, 0) << "y:" << last_T(1, 0)
                  << "z:" << last_T(2, 0) << std::endl;

        if (coarseTracker->firstCoarseRMSE < 0)
            coarseTracker->firstCoarseRMSE = achievedRes[0];

        if (!setting_debugout_runquiet)
            printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a,
                   aff_g2l.b, fh->ab_exposure, achievedRes[0]);

        if (setting_logStuff) {
            (*coarseTrackingLog) << std::setprecision(16) << fh->shell->id << " "
                                 << fh->shell->timestamp << " " << fh->ab_exposure
                                 << " " << fh->shell->camToWorld.log().transpose()
                                 << " " << aff_g2l.a << " " << aff_g2l.b << " "
                                 << achievedRes[0] << " " << tryIterations << "\n";
        }

        return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
    }

//======计算双目深度======
    void FullSystem::computeIdepth(FrameHessian *fh, FrameHessian *fh_right) {

        //    makeNewTraces(fh, 0);

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib.fxl();
        K(1, 1) = Hcalib.fyl();
        K(0, 2) = Hcalib.cxl();
        K(1, 2) = Hcalib.cyl();

        // KRKi为单位矩阵
        Mat33f KRKi = Mat33f::Identity().cast<float>();
        // Kt为baseline平移矩阵
        Vec3f bl;
        Vec3f Kt;
        bl << -0.53716572, 0, 0;
        Kt = K * bl;
        //简化情况的aff
        Vec2f aff;
        aff << 1, 0;

        float bf = -K(0, 0) * bl[0];

        int lvl = 0;
        float depth = 0;
        float idepth = 0;
        float u = 0;
        float v = 0;
        float match_u = 0;
        float match_v = 0;
        float scale = 0;

        bool debugPrintMatchPoint = false;
        bool debugPrintDepthImage = false;

        //======Show matched points between left and right images======
        if (debugPrintMatchPoint) {
            MinimalImageB3 mf(wG[lvl], 2 * hG[lvl]);
            mf.setBlack();

            for (int i = 0; i < hG[lvl] * wG[lvl]; i++) {
                int c = fh->dIp[lvl][i][0] * 0.9f;
                if (c > 255)
                    c = 255;
                mf.at(i) = Vec3b(c, c, c);
            }
            for (int i = hG[lvl] * wG[lvl]; i < 2 * hG[lvl] * wG[lvl]; i++) {
                int c = fh_right->dIp[lvl][i - hG[lvl] * wG[lvl]][0];
                if (c > 255)
                    c = 255;
                mf.at(i) = Vec3b(c, c, c);
            }
            cv::Mat image(2 * hG[lvl], wG[lvl], CV_8UC3, mf.data);

            cv::Point left_temp;
            cv::Point right_temp;
            int counter = 0;
            int number = 0;

            for (ImmaturePoint *ph : fh->immaturePoints) {
                ph->traceOn(fh_right, KRKi, Kt, aff, &Hcalib, false);
                u = ph->u;
                v = ph->v;
                match_u = ph->lastTraceUV[0];
                match_v = ph->lastTraceUV[1];

                counter++;
                if (match_u != -1 && match_v != -1) {

                    number++;
                    if (counter % 4 == 0) {
                        left_temp.x = u;
                        left_temp.y = v;
                        right_temp.x = match_u;
                        right_temp.y = hG[lvl] + match_v;
                        cv::circle(image, left_temp, 6, (255, 0, 0), 2);
                        cv::circle(image, right_temp, 6, (255, 0, 0), 2);
                        cv::line(image, left_temp, right_temp, (0, 0, 255), 2);
                    }
                }
            }

            cv::imshow("matched image", image);
            printf("the number of good points is %d \n", number);
        }

        //======Show depth map======
        if (debugPrintDepthImage) {

            MinimalImageB3 mf(wG[lvl], hG[lvl]);
            mf.setBlack();
            for (ImmaturePoint *ph : fh->immaturePoints) {
                ph->traceOn(fh_right, KRKi, bl, aff, &Hcalib, false);
                u = ph->u;
                v = ph->v;
                match_u = ph->lastTraceUV[0];
                match_v = ph->lastTraceUV[1];
                if ((match_u != -1) && (match_v != -1)) {
                    depth = bf / (u - match_u);
                    ph->idepth_min = 1.0f / depth;
                    ph->idepth_max = 1.0f / depth;
                    if (depth < 10) {
                        mf.at(ph->u, ph->v) = Vec3b(0, 0, 255);
                        mf.at(ph->u - 1, ph->v - 1) = Vec3b(0, 0, 255);
                        mf.at(ph->u - 1, ph->v + 1) = Vec3b(0, 0, 255);
                        mf.at(ph->u + 1, ph->v + 1) = Vec3b(0, 0, 255);
                        mf.at(ph->u + 1, ph->v - 1) = Vec3b(0, 0, 255);

                    } else if (depth < 20) {
                        mf.at(ph->u, ph->v) = Vec3b(0, 255, 0);
                        mf.at(ph->u - 1, ph->v - 1) = Vec3b(0, 255, 0);
                        mf.at(ph->u - 1, ph->v + 1) = Vec3b(0, 255, 0);
                        mf.at(ph->u + 1, ph->v + 1) = Vec3b(0, 255, 0);
                        mf.at(ph->u + 1, ph->v - 1) = Vec3b(0, 255, 0);
                    } else if (depth < 40) {
                        mf.at(ph->u, ph->v) = Vec3b(255, 255, 0);
                        mf.at(ph->u - 1, ph->v - 1) = Vec3b(255, 255, 0);
                        mf.at(ph->u - 1, ph->v + 1) = Vec3b(255, 255, 0);
                        mf.at(ph->u + 1, ph->v + 1) = Vec3b(255, 255, 0);
                        mf.at(ph->u + 1, ph->v - 1) = Vec3b(255, 255, 0);
                    } else {
                        mf.at(ph->u, ph->v) = Vec3b(255, 0, 0);
                        mf.at(ph->u - 1, ph->v - 1) = Vec3b(255, 0, 0);
                        mf.at(ph->u - 1, ph->v + 1) = Vec3b(255, 0, 0);
                        mf.at(ph->u + 1, ph->v + 1) = Vec3b(255, 0, 0);
                        mf.at(ph->u + 1, ph->v - 1) = Vec3b(255, 0, 0);
                    }

                    printf("u is %f and the matched u is %f, depth is %f \n", u, match_u,
                           depth);
                    printf("v is %f and the matched v is %f, depth is %f \n", v, match_v,
                           depth);
                }
            }
            IOWrap::displayImage("stereo depth map left", &mf);
        }

        //======Compute Idepth======
        for (ImmaturePoint *ph : fh->immaturePoints) {
            ImmaturePointStatus stat =
                    ph->traceOn(fh_right, KRKi, bl, aff, &Hcalib, false);
            u = ph->u;
            v = ph->v;
            match_u = ph->lastTraceUV[0];
            match_v = ph->lastTraceUV[1];
            if ((match_u != -1) && (match_v != -1) && stat == IPS_GOOD) {
                depth = bf / (u - match_u);
                idepth = 1.0f / depth;
                scale = (ph->idepth_min + ph->idepth_max) * 0.5f / idepth;
                ph->idepth_stereo = idepth;
                ph->idepth_min = idepth;
                ph->idepth_max = idepth;
                //			printf("u is %f and the matched u is %f, depth
                // is %f \n", u, match_u, depth);
                //			printf("v is %f and the matched v is %f, depth
                // is %f \n", v, match_v, depth);
                //          printf("scale is %f, stereo depth is %f, idepth_min is %f,
                //          idepth_max is %f \n", scale, depth, ph->idepth_min,
                //          ph->idepth_max);
            } else {
                ph->idepth_stereo = NAN;
            }
        }
    }

    void FullSystem::traceNewCoarseNonKey(FrameHessian *fh,
                                          FrameHessian *fh_right) {

        boost::unique_lock<boost::mutex> lock(mapMutex);

        int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
                trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

        float idepth_track = 0; // track得到深度
        float idepth_update = 0; // target帧stereo得到深度后inverse得到的host帧深度
        float idepth_min_update =
                0; // target帧stereo得到idpeth_min_stereo后inverse得到的host帧idepth_min
        float idepth_max_update =
                0; // target帧stereo得到idpeth_max_stereo后inverse得到的host帧idepth_max
        float conTrack = 0;  // track得到的深度的置信度
        float conStereo = 0; // stereo得到的深度的置信度

        float idepth_min_temp = 0;
        float idepth_max_temp = 0;

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib.fxl();
        K(1, 1) = Hcalib.fyl();
        K(0, 2) = Hcalib.cxl();
        K(1, 2) = Hcalib.cyl();

        float bf = K(0, 0) * 0.53716572;

        Mat33f Ki = K.inverse();

        /*		int lvl = 0;
                        MinimalImageB3 mf(wG[lvl], 2*hG[lvl]);
                        mf.setBlack();

                        for(int i=0;i<hG[lvl]*wG[lvl];i++)
                        {
                                int c = fh->dIp[lvl][i][0]*0.9f;
                                if(c>255) c=255;
                                mf.at(i) = Vec3b(c,c,c);
                        }
                        for(int i=hG[lvl]*wG[lvl];i<2*hG[lvl]*wG[lvl];i++)
                        {
                                int c = fh_right->dIp[lvl][i-hG[lvl]*wG[lvl]][0];
                                if(c>255) c=255;
                                mf.at(i) = Vec3b(c,c,c);
                        }

                        cv::Mat image(2*hG[lvl], wG[lvl], CV_8UC3, mf.data);

                        cv::Point left_temp;
                        cv::Point right_temp;
                        int counter = 0;
                int number = 0;*/

        //======Read disparity pics from SGM======
        int id = (int) allFrameHistory.size() - 1;
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(6) << id;
        std::string disparityPath =
                "/home/jiatianwu/dso/05/disparity/" + ss.str() + ".png";
        cv::Mat m = cv::imread(disparityPath);
        MinimalImageB *disparity = new MinimalImageB(m.cols, m.rows);
        memcpy(disparity->data, m.data, m.rows * m.cols);

        //对于所有active的frame
        for (FrameHessian *host : frameHessians) // go through all active frames
        {

            //            number ++ ;

            //参考帧到最新一帧的旋转平移
            SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
            // KRK-1
            Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();

            Mat33f RiKi =
                    hostToNew.rotationMatrix().inverse().cast<float>() * K.inverse();

            Mat33f KR = K * hostToNew.rotationMatrix().cast<float>();
            Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
            Mat33f Ri = hostToNew.rotationMatrix().inverse().cast<float>();
            // Kt
            Vec3f Kt = K * hostToNew.translation().cast<float>();
            Vec3f t = hostToNew.translation().cast<float>();

            // aff
            Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure,
                                                    host->aff_g2l(), fh->aff_g2l())
                    .cast<float>();

            //对于当前帧host中的每一个点
            for (ImmaturePoint *ph : host->immaturePoints) {
                //在里面会求点的深度
                ImmaturePointStatus phTrackStatus =
                        ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

                if (phTrackStatus == ImmaturePointStatus::IPS_GOOD) {
                    idepth_min_temp = ph->idepth_min;
                    idepth_max_temp = ph->idepth_max;
                    idepth_track = (idepth_min_temp + idepth_max_temp) * 0.5f;
                    float weight_min = idepth_min_temp / idepth_track;
                    float weight_max = idepth_max_temp / idepth_track;
                    //得到track后的深度和它的置信度
                    /*					idepth_track = (idepth_min_temp
                       + idepth_max_temp) * 0.5f; conTrack = (idepth_max_temp -
                       idepth_min_temp) * (idepth_max_temp - idepth_min_temp);

                                                            Vec3f ptpMin = KRKi *
                       (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt; float
                       idepth_min_project = 1.0f / ptpMin[2]; Vec3f ptpMax = KRKi *
                       (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt; float
                       idepth_max_project = 1.0f / ptpMax[2];

                                                            Vec3f ptpMid = KRKi * (2 *
                       Vec3f(ph->u, ph->v, 1) / (ph->idepth_max + ph->idepth_min)) + Kt;
                                                            float u_project = ptpMid[0] /
                       ptpMid[2]; float v_project = ptpMid[1] / ptpMid[2];*/

                    ImmaturePoint *phNonKey = new ImmaturePoint(
                            ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

                    float sgmDisparity =
                            getMatInterpolatedElement11BiLin(m, phNonKey->u, phNonKey->v);

                    if (sgmDisparity < 20) {
                        ph->lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
                        continue;
                    }

                    float sgm_idepth_min = weight_min * (sgmDisparity / bf);
                    float sgm_idepth_max = weight_max * (sgmDisparity / bf);

                    //                    printf("dispariy is %f  %f %f \n", sgmDisparity,
                    //                    sgm_idepth_min, sgm_idepth_max);

                    Vec3f pinverse_min =
                            KRi *
                            (Ki * Vec3f(phNonKey->u, phNonKey->v, 1) / sgm_idepth_min - t);
                    idepth_min_update = 1.0f / pinverse_min(2);

                    Vec3f pinverse_max =
                            KRi *
                            (Ki * Vec3f(phNonKey->u, phNonKey->v, 1) / sgm_idepth_max - t);
                    idepth_max_update = 1.0f / pinverse_max(2);

                    ph->idepth_min = idepth_min_update;
                    ph->idepth_max = idepth_max_update;

                    //					if(host == frameHessians.back())
                    //					{
                    //						printf(" min %f max %f min new
                    //%f  max new %f \n", idepth_min_temp, idepth_max_temp,
                    // idepth_min_update,  idepth_max_update);
                    //					}

                    /*                    counter++;

                                        if(counter%4 == 0)
                                        {
                                            left_temp.x = phNonKey->u; left_temp.y =
                       phNonKey->v; right_temp.x = phNonKey->u - sgmDisparity; right_temp.y
                       = hG[lvl] + phNonKey->v; cv::circle(image, left_temp, 6, (255,0,0),
                       2); cv::circle(image, right_temp, 6, (255,0,0), 2); cv::line(image,
                       left_temp, right_temp,(0,0,255), 2);
                                        }*/
                }

                /*					phNonKey->idepth_min =
                idepth_min_project; phNonKey->idepth_max = idepth_max_project;
                                                        phNonKey->u_stereo = phNonKey->u;
                                                        phNonKey->v_stereo = phNonKey->v;

                                    if (phNonKey->idepth_min < 0.01 ||
                phNonKey->idepth_max < 0.01 || phNonKey->idepth_min > 0.1 ||
                phNonKey->idepth_max > 0.1)
                                    {

                                                        phNonKey->idepth_min_stereo = 0;
                                                        phNonKey->idepth_max_stereo = NAN;

                                    } else{

                                        phNonKey->idepth_min_stereo =
                phNonKey->idepth_min; phNonKey->idepth_max_stereo = phNonKey->idepth_max;

                                    }


                //                    printf(" u %f v %f pu %f pv %f \n", phNonKey->u,
                phNonKey->v, u_project, v_project);
                //                    printf(" %f %f project min %f project max %f \n",
                ph->idepth_min, ph->idepth_max, phNonKey->idepth_min,
                phNonKey->idepth_max);

                                                        ImmaturePointStatus
                phNonKeyStereoStatus = phNonKey->traceright(fh_right, K);

                                                        if (phNonKeyStereoStatus !=
                ImmaturePointStatus::IPS_GOOD)
                                                        {
                                                                continue;

                                                        }
                                                        else
                                                        {

                                                                Vec3f pinverse_min = KRi *
                (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) /
                phNonKey->idepth_min_stereo - t); idepth_min_update = 1.0f /
                pinverse_min(2);

                                                                Vec3f pinverse_max = KRi *
                (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) /
                phNonKey->idepth_max_stereo - t); idepth_max_update = 1.0f /
                pinverse_max(2);

                                                                ph->idepth_min =
                idepth_min_update; ph->idepth_max = idepth_max_update;



                                                        printf("o min %f o max %f p min %f
                p max %f s min %f s max %f rp min %f rp max %f \n", idepth_min_temp,
                idepth_max_temp, idepth_min_project, idepth_max_project,
                phNonKey->idepth_min_stereo, phNonKey->idepth_max_stereo, ph->idepth_min,
                ph->idepth_max);

                                                                Vec3f pinverse_mid = KRi *
                (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) /
                phNonKey->idepth_stereo - t); float u_update = pinverse_mid[0] /
                pinverse_mid[2]; float v_update = pinverse_mid[1] / pinverse_mid[2];

                                                                printf("%f %f %f %f %f
                \n", u_update, v_update, phNonKey->idepth_min_stereo,
                phNonKey->idepth_max_stereo, phNonKey->idepth_stereo); printf("min %f max
                %f new min %f new max %f \n", idepth_min_temp, idepth_max_temp,
                ph->idepth_min, ph->idepth_max);


                                                        }*/
            }
            /*			if(allFrameHistory.size() > 0)
                                    {

            //				cv::imshow("matched image", image);

                                            std::stringstream stream1;
                                            std::string str1;
                                            stream1<<allFrameHistory.size()-1;
                                            stream1>>str1;

                                            std::stringstream stream2;
                                            std::string str2;
                                            stream2<<number;
                                            stream2>>str2;

                                            std::string str = str1 + str2;

                                            cv::imwrite("/home/jiatianwu/dso/sdso_latest/match/"
            + str + ".png", image);

            //				cvWaitKey(0.01);

                                    }*/
            //			printf("the number of good points is %d \n", counter);
        }
    }

    void FullSystem::traceNewCoarseNonKeyWithMatchPic(FrameHessian *fh,
                                                      FrameHessian *fh_right) {
        boost::unique_lock<boost::mutex> lock(mapMutex);

        float idepth_track = 0; // track得到深度
        float idepth_update = 0; // target帧stereo得到深度后inverse得到的host帧深度
        float idepth_min_update =
                0; // target帧stereo得到idpeth_min_stereo后inverse得到的host帧idepth_min
        float idepth_max_update =
                0; // target帧stereo得到idpeth_max_stereo后inverse得到的host帧idepth_max
        float conTrack = 0;  // track得到的深度的置信度
        float conStereo = 0; // stereo得到的深度的置信度

        float idepth_min_temp = 0;
        float idepth_max_temp = 0;

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib.fxl();
        K(1, 1) = Hcalib.fyl();
        K(0, 2) = Hcalib.cxl();
        K(1, 2) = Hcalib.cyl();

        Mat33f Ki = K.inverse();

        /*	int lvl = 0;
                MinimalImageB3 mf(wG[lvl], 2 * hG[lvl]);
                mf.setBlack();

                for (int i = 0; i < hG[lvl] * wG[lvl]; i++) {
                        int c = fh->dIp[lvl][i][0] * 0.9f;
                        if (c > 255) c = 255;
                        mf.at(i) = Vec3b(c, c, c);
                }
                for (int i = hG[lvl] * wG[lvl]; i < 2 * hG[lvl] * wG[lvl]; i++) {
                        int c = fh_right->dIp[lvl][i - hG[lvl] * wG[lvl]][0];
                        if (c > 255) c = 255;
                        mf.at(i) = Vec3b(c, c, c);
                }

                cv::Mat image(2 * hG[lvl], wG[lvl], CV_8UC3, mf.data);

                cv::Point left_temp;
                cv::Point right_temp;
                int counter = 0;
                int number = 0;*/

        //对于所有active的frame
        for (FrameHessian *host : frameHessians) // go through all active frames
        {

            //		number++;
            int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
                    trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

            //参考帧到最新一帧的旋转平移
            SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
            // KRK-1
            Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();

            Mat33f RiKi =
                    hostToNew.rotationMatrix().inverse().cast<float>() * K.inverse();

            Mat33f KR = K * hostToNew.rotationMatrix().cast<float>();
            Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
            Mat33f Ri = hostToNew.rotationMatrix().inverse().cast<float>();
            // Kt
            Vec3f Kt = K * hostToNew.translation().cast<float>();
            Vec3f t = hostToNew.translation().cast<float>();

            // aff
            Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure,
                                                    host->aff_g2l(), fh->aff_g2l())
                    .cast<float>();

            int allImmaturePoints = host->immaturePoints.size();
            int stereoBAImmaturePoints = 0;
            int stereoGoodImmaturePoints = 0;

            //对于当前帧host中的每一个点
            for (ImmaturePoint *ph : host->immaturePoints) {
                //在里面会求点的深度
                ImmaturePointStatus phTrackStatus =
                        ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

                if (phTrackStatus == ImmaturePointStatus::IPS_GOOD) {
                    idepth_min_temp = ph->idepth_min;
                    idepth_max_temp = ph->idepth_max;
                    //得到track后的深度和它的置信度
                    idepth_track = (idepth_min_temp + idepth_max_temp) * 0.5f;
                    conTrack = (idepth_max_temp - idepth_min_temp) *
                               (idepth_max_temp - idepth_min_temp);

                    ImmaturePoint *phNonKey = new ImmaturePoint(
                            ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

                    Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
                    float idepth_min_project = 1.0f / ptpMin[2];
                    Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
                    float idepth_max_project = 1.0f / ptpMax[2];

                    Vec3f ptpMid = KRKi * (2 * Vec3f(ph->u, ph->v, 1) /
                                           (ph->idepth_max + ph->idepth_min)) +
                                   Kt;
                    float u_project = ptpMid[0] / ptpMid[2];
                    float v_project = ptpMid[1] / ptpMid[2];

                    phNonKey->idepth_min = idepth_min_project;
                    phNonKey->idepth_max = idepth_max_project;
                    phNonKey->u_stereo = phNonKey->u;
                    phNonKey->v_stereo = phNonKey->v;

                    phNonKey->idepth_min_stereo = 0;
                    phNonKey->idepth_max_stereo = NAN;

                    ImmaturePointStatus phNonKeyStereoStatus =
                            phNonKey->traceright(fh_right, K);

                    ImmaturePoint *phNonKeyRight =
                            new ImmaturePoint(phNonKey->lastTraceUV(0),
                                              phNonKey->lastTraceUV(1), fh_right, &Hcalib);

                    phNonKeyRight->u_stereo = phNonKeyRight->u;
                    phNonKeyRight->v_stereo = phNonKeyRight->v;
                    phNonKeyRight->idepth_min_stereo = 0;
                    phNonKeyRight->idepth_max_stereo = NAN;
                    ImmaturePointStatus phNonKeyRightStereoStatus =
                            phNonKeyRight->traceleft(fh, K);

                    float u_stereo_delta =
                            abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
                    //				std::cout<< u_stereo_delta << std::endl;
                    //				printf(" %f %f \n", phNonKey->u_stereo,
                    // phNonKeyRight->lastTraceUV(0));
                    if (u_stereo_delta > 1) {
                        ph->lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
                        continue;
                    }
                    stereoBAImmaturePoints++;

                    float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

                    if (disparity < 10) {
                        ph->lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
                        continue;
                    }

                    if (phNonKeyStereoStatus != ImmaturePointStatus::IPS_GOOD) {
                        continue;

                    } else {

                        stereoGoodImmaturePoints++;
                        Vec3f pinverse_min =
                                KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) /
                                       phNonKey->idepth_min_stereo -
                                       t);
                        idepth_min_update = 1.0f / pinverse_min(2);

                        Vec3f pinverse_max =
                                KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) /
                                       phNonKey->idepth_max_stereo -
                                       t);
                        idepth_max_update = 1.0f / pinverse_max(2);

                        ph->idepth_min = idepth_min_update;
                        ph->idepth_max = idepth_max_update;
                    }

                    delete phNonKey;
                    delete phNonKeyRight;

                    /*				counter++;

                                                    if (counter % 8 == 0) {
                                                            left_temp.x = phNonKey->u;
                                                            left_temp.y = phNonKey->v;
                                                            right_temp.x =
                       phNonKey->lastTraceUV[0]; right_temp.y = hG[lvl] +
                       phNonKey->lastTraceUV[1]; cv::circle(image, left_temp, 6, (255, 0,
                       0), 2); cv::circle(image, right_temp, 6, (255, 0, 0), 2);
                                                            cv::line(image, left_temp,
                       right_temp, (0, 0, 255), 2);
                                                    }*/
                }

                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
                    trace_good++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
                    trace_badcondition++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
                    trace_oob++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                    trace_out++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
                    trace_skip++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
                    trace_uninitialized++;
                trace_total++;
            }

            double BApercent =
                    (allImmaturePoints == 0)
                    ? 0
                    : (double) stereoBAImmaturePoints / (double) allImmaturePoints;
            double Goodpercent =
                    (allImmaturePoints == 0)
                    ? 0
                    : (double) stereoGoodImmaturePoints / (double) allImmaturePoints;
            //		printf("all points %d  track good %d BA points %f good points %f
            //\n",  allImmaturePoints, trace_good, BApercent, Goodpercent);
            /*		if (allFrameHistory.size() > 700) {

            //			cv::imshow("matched image", image);

                                    std::stringstream stream1;
                                    std::string str1;
                                    stream1 << allFrameHistory.size();
                                    stream1 >> str1;

                                    std::stringstream stream2;
                                    std::string str2;
                                    stream2 << number;
                                    stream2 >> str2;

                                    std::string str = str1 + str2;

                                    cv::imwrite("/home/jiatianwu/dso/sdso_latest/match/"
            + str + ".png", image);

            //			cvWaitKey(0.01);

                            }
                            printf("the number of good points is %d \n", counter);*/
        }
    }



    /**
    * 跟踪当前帧fh中的的点
    * 遍历帧待处理队列frameHessian(vector<FrameHessian *>),遍历帧中的每个immature点，
         * 使用函数traceOn标记这些点的状态，然后统计不同状态下点的数量。
         * 无论是否keyframe，都要trace一下。加固每一个immaturePoint。
    * @param fh 当前处理帧
    * @return
    */

    void FullSystem::traceNewCoarse(FrameHessian *fh, FrameHessian *fh_right) {
        boost::unique_lock<boost::mutex> lock(mapMutex);

        int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0,
                trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

        float idepth_track = 0; // track得到深度
        float idepth_update = 0; // target帧stereo得到深度后inverse得到的host帧深度
        float idepth_min_update =
                0; // target帧stereo得到idpeth_min_stereo后inverse得到的host帧idepth_min
        float idepth_max_update =
                0; // target帧stereo得到idpeth_max_stereo后inverse得到的host帧idepth_max
        float conTrack = 0;  // track得到的深度的置信度
        float conStereo = 0; // stereo得到的深度的置信度

        float idepth_min_temp = 0;
        float idepth_max_temp = 0;
        float scale = 0;

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib.fxl();
        K(1, 1) = Hcalib.fyl();
        K(0, 2) = Hcalib.cxl();
        K(1, 2) = Hcalib.cyl();

        //对于所有active的frame
        for (FrameHessian *host : frameHessians) // go through all active frames
        {

            //参考帧到最新一帧的旋转平移
            SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
            // KRK-1
            Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
            Mat33f RiKi =
                    hostToNew.rotationMatrix().inverse().cast<float>() * K.inverse();
            Mat33f KR = K * hostToNew.rotationMatrix().cast<float>();
            // Kt
            Vec3f Kt = K * hostToNew.translation().cast<float>();

            Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure,
                                                    host->aff_g2l(), fh->aff_g2l())
                    .cast<float>();

            //对于当前帧host中的每一个点
            for (ImmaturePoint *ph : host->immaturePoints) {
                //在里面会求点的深度
                ImmaturePointStatus phTrackStatus =
                        ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

                /*				if(phTrackStatus ==
                ImmaturePointStatus::IPS_GOOD)
                                                {
                                                        idepth_min_temp = ph->idepth_min;
                                                        idepth_max_temp = ph->idepth_max;
                                                        //得到track后的深度和它的置信度
                                                        idepth_track = (idepth_min_temp +
                idepth_max_temp)*0.5f; conTrack =
                (idepth_max_temp-idepth_min_temp)*(idepth_max_temp-idepth_min_temp);

                                                        ph->u_stereo = ph->lastTraceUV(0);
                                                        ph->v_stereo = ph->lastTraceUV(1);
                                    ph->idepth_min_stereo = ph->idepth_min;
                                    ph->idepth_max_stereo = ph->idepth_max;
                //		            ph->idepth_min_stereo = 0;
                //		            ph->idepth_max_stereo = NAN;
                                                        ImmaturePointStatus phStereoStatus
                = ph->traceright(fh_right, K);

                //					if(ph->idepth_stereo!=0 && ph->idepth_min > 0
                && ph->idepth_max > 0) if(phStereoStatus == ImmaturePointStatus ::
                IPS_GOOD)
                                                        {
                                                                Vec3f pinverse =
                RiKi*(Vec3f(ph->u_stereo, ph->v_stereo, 1)/ph->idepth_stereo - Kt);
                                                                idepth_update
                = 1.0f/pinverse(2);

                                        Vec3f pinverse_min = RiKi*(Vec3f(ph->u_stereo,
                ph->v_stereo, 1)/ph->idepth_min_stereo - Kt); idepth_min_update
                = 1.0f/pinverse_min(2);

                                        Vec3f pinverse_max = RiKi*(Vec3f(ph->u_stereo,
                ph->v_stereo, 1)/ph->idepth_max_stereo - Kt); idepth_max_update
                = 1.0f/pinverse_max(2);

                                        conStereo = (idepth_max_update -
                idepth_min_update)*(idepth_max_update - idepth_min_update);

                //						ph->idepth_min = (conStereo*idepth_track
                + conTrack*idepth_update - conStereo*conTrack)/(conStereo + conTrack);
                //						ph->idepth_max = (conStereo*idepth_track
                + conTrack*idepth_update + conStereo*conTrack)/(conStereo + conTrack);

                                        ph->idepth_min = idepth_min_update;
                                        ph->idepth_max = idepth_max_update;

                //                        ph->idepth_min = (idepth_update -
                (ph->idepth_max_stereo - ph->idepth_min_stereo)*0.5f)*0.5f +
                idepth_min_temp*0.5f;
                //                        ph->idepth_max = (idepth_update +
                (ph->idepth_max_stereo - ph->idepth_min_stereo)*0.5f)*0.5f +
                idepth_max_temp*0.5f;
                //					    scale =
                (idepth_min_temp+idepth_max_temp)*0.5f/idepth_update;
                //					    printf("min %f min update %f max %f max update %f
                \n", idepth_min_temp, ph->idepth_min, idepth_max_temp, ph->idepth_max);
                                                        }

                                                }*/

                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
                    trace_good++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
                    trace_badcondition++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
                    trace_oob++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                    trace_out++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
                    trace_skip++;
                if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
                    trace_uninitialized++;
                trace_total++;
            }
        }
    }

// 看看active
// frame的immaturePoints能不能在当前帧被track到，并更新这些points的状态
/*void FullSystem::traceNewCoarse(FrameHessian* fh, FrameHessian* fh_right)
{
        boost::unique_lock<boost::mutex> lock(mapMutex);

        int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0,
trace_badcondition=0, trace_uninitialized=0;

        Mat33f K = Mat33f::Identity();
        K(0,0) = Hcalib.fxl();
        K(1,1) = Hcalib.fyl();
        K(0,2) = Hcalib.cxl();
        K(1,2) = Hcalib.cyl();

            //KRKi为单位矩阵
    Mat33f KRKi = Mat33f::Identity();
    //Kt为baseline
    Vec3f bl;
    bl << 0.53716572, 0, 0;
    Vec3f Kt = K * bl;
    //简化情况的aff
    Vec2f aff;
    aff << 1, 0;

    float bf = K(0,0)*bl[0];

        //对于所有active的frame
        for(FrameHessian* host : frameHessians)		// go through all
active frames
        {

                //参考帧到最新一帧的旋转平移
                SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
                //KRK-1
                Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() *
K.inverse();
                //Kt
                Vec3f Kt = K * hostToNew.translation().cast<float>();

                Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure,
fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

                //对于当前帧host中的每一个点
                for(ImmaturePoint* ph : host->immaturePoints)
                {
                        //在里面会求点的深度
                        ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );
                                    if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD)
trace_good++; if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION)
trace_badcondition++; if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB)
trace_oob++; if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER)
trace_out++; if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED)
trace_skip++; if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED)
trace_uninitialized++; trace_total++;
                }
        }
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip.
%'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%)
uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition,
100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized,
100*trace_uninitialized/(float)trace_total);
}*/

    void FullSystem::activatePointsMT_Reductor(
            std::vector<PointHessian *> *optimized,
            std::vector<ImmaturePoint *> *toOptimize, int min, int max, Vec10 *stats,
            int tid) {
        ImmaturePointTemporaryResidual *tr =
                new ImmaturePointTemporaryResidual[frameHessians.size()];
        for (int k = min; k < max; k++) {
            (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
        }
        delete[] tr;
    }

    void FullSystem::activatePointsMT() {

        if (ef->nPoints <
            setting_desiredPointDensity * 0.66) // setting_desiredPointDensity 是2000
            currentMinActDist -= 0.8;             // original 0.8
        if (ef->nPoints < setting_desiredPointDensity * 0.8)
            currentMinActDist -= 0.5; // original 0.5
        else if (ef->nPoints < setting_desiredPointDensity * 0.9)
            currentMinActDist -= 0.2; // original 0.2
        else if (ef->nPoints < setting_desiredPointDensity)
            currentMinActDist -= 0.1; // original 0.1

        if (ef->nPoints > setting_desiredPointDensity * 1.5)
            currentMinActDist += 0.8;
        if (ef->nPoints > setting_desiredPointDensity * 1.3)
            currentMinActDist += 0.5;
        if (ef->nPoints > setting_desiredPointDensity * 1.15)
            currentMinActDist += 0.2;
        if (ef->nPoints > setting_desiredPointDensity)
            currentMinActDist += 0.1;

        if (currentMinActDist < 0)
            currentMinActDist = 0;
        if (currentMinActDist > 4)
            currentMinActDist = 4;

        if (!setting_debugout_runquiet)
            printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                   currentMinActDist, (int) (setting_desiredPointDensity), ef->nPoints);

        FrameHessian *newestHs = frameHessians.back();

        // make dist map.
        coarseDistanceMap->makeK(&Hcalib);
        coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

        // coarseTracker->debugPlotDistMap("distMap");

        std::vector<ImmaturePoint *> toOptimize;
        toOptimize.reserve(20000);

        for (FrameHessian *host : frameHessians) // go through all active frames
        {
            if (host == newestHs)
                continue;

            SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
            Mat33f KRKi =
                    (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() *
                     coarseDistanceMap->Ki[0]);
            Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

            //对frameHessian中的immaturePoint进行操作
            for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1) {
                ImmaturePoint *ph = host->immaturePoints[i];
                ph->idxInImmaturePoints = i;

                // delete points that have never been traced successfully, or that are
                // outlier on the last trace.
                if (!std::isfinite(ph->idepth_max) ||
                    ph->lastTraceStatus == IPS_OUTLIER) {
                    //				immature_invalid_deleted++;
                    // remove point.
                    delete ph;
                    host->immaturePoints[i] = 0;
                    continue;
                }

                // can activate only if this is true.
                bool canActivate = (ph->lastTraceStatus == IPS_GOOD ||
                                    ph->lastTraceStatus == IPS_SKIPPED ||
                                    ph->lastTraceStatus == IPS_BADCONDITION ||
                                    ph->lastTraceStatus == IPS_OOB) &&
                                   ph->lastTracePixelInterval < 8 &&
                                   ph->quality > setting_minTraceQuality &&
                                   (ph->idepth_max + ph->idepth_min) > 0;

                // if I cannot activate the point, skip it. Maybe also delete it.
                if (!canActivate) {
                    // if point will be out afterwards, delete it instead.
                    if (ph->host->flaggedForMarginalization ||
                        ph->lastTraceStatus == IPS_OOB) {
                        //					immature_notReady_deleted++;
                        delete ph;
                        host->immaturePoints[i] = 0;
                    }
                    //				immature_notReady_skipped++;
                    continue;
                }

                // see if we need to activate point due to distance map.
                Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) +
                            Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
                int u = ptp[0] / ptp[2] + 0.5f;
                int v = ptp[1] / ptp[2] + 0.5f;

                if ((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {

                    float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] +
                                 (ptp[0] - floorf((float) (ptp[0])));

                    if (dist >= currentMinActDist * ph->my_type) {
                        coarseDistanceMap->addIntoDistFinal(u, v);
                        toOptimize.push_back(ph);
                    }
                } else {
                    delete ph;
                    host->immaturePoints[i] = 0; //删除点的操作
                }
            }
        }

        //	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip
        //%d)\n", 			(int)toOptimize.size(), immature_deleted,
        // immature_notReady, immature_needMarg, immature_want, immature_margskip);

        std::vector<PointHessian *> optimized;
        optimized.resize(toOptimize.size());

        if (multiThreading)
            treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this,
                                           &optimized, &toOptimize, _1, _2, _3, _4),
                               0, toOptimize.size(), 50);

        else
            activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0,
                                      0);

        for (unsigned k = 0; k < toOptimize.size(); k++) {
            PointHessian *newpoint = optimized[k];
            ImmaturePoint *ph = toOptimize[k];

            if (newpoint != 0 && newpoint != (PointHessian *) ((long) (-1))) {
                newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
                newpoint->host->pointHessians.push_back(newpoint);
                ef->insertPoint(newpoint);
                for (PointFrameResidual *r : newpoint->residuals)
                    ef->insertResidual(r);
                assert(newpoint->efPoint != 0);
                delete ph;
            } else if (newpoint == (PointHessian *) ((long) (-1)) ||
                       ph->lastTraceStatus == IPS_OOB) {
                delete ph;
                ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
            } else {
                assert(newpoint == 0 || newpoint == (PointHessian *) ((long) (-1)));
            }
        }

        for (FrameHessian *host : frameHessians) {
            for (int i = 0; i < (int) host->immaturePoints.size(); i++) {
                if (host->immaturePoints[i] == 0) {
                    host->immaturePoints[i] = host->immaturePoints.back();
                    host->immaturePoints.pop_back();
                    i--;
                }
            }
        }
    }

    void FullSystem::activatePointsOldFirst() { assert(false); }
    /**
    * 标记需要移除和边缘化的点
     * 1. 遍历待处理帧队列
     * 2. 如果当前帧编辑为待边缘化的帧，出入一个容器
     * 3. 遍历容器中的每个帧
     *      遍历帧中的每个immature point，
     *          如果该点尺度小于0或者残差数量为0 插入Hessianout容器
     *          否则 并且是内点 该点内的每个残差进行线性化、apply res。
     *
    * @param
    * @return
    */
    void FullSystem::flagPointsForRemoval() {
        assert(EFIndicesValid);

        std::vector<FrameHessian *> fhsToKeepPoints;
        std::vector<FrameHessian *> fhsToMargPoints;

        // if(setting_margPointVisWindow>0)
        {
            for (int i = ((int) frameHessians.size()) - 1;
                 i >= 0 && i >= ((int) frameHessians.size()); i--)
                if (!frameHessians[i]->flaggedForMarginalization)
                    fhsToKeepPoints.push_back(frameHessians[i]);

            for (int i = 0; i < (int) frameHessians.size(); i++)
                if (frameHessians[i]->flaggedForMarginalization)
                    fhsToMargPoints.push_back(frameHessians[i]);
        }

        // ef->setAdjointsF();
        // ef->setDeltaF(&Hcalib);
        int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

        for (FrameHessian *host : frameHessians) // go through all active frames
        {
            for (unsigned int i = 0; i < host->pointHessians.size(); i++) {
                PointHessian *ph = host->pointHessians[i];
                if (ph == 0)
                    continue;

                if (ph->idepth_scaled < 0 || ph->residuals.size() == 0) {
                    host->pointHessiansOut.push_back(ph);
                    ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                    host->pointHessians[i] = 0;
                    flag_nores++;
                } else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) ||
                           host->flaggedForMarginalization) {
                    flag_oob++;
                    if (ph->isInlierNew()) {
                        flag_in++;
                        int ngoodRes = 0;
                        for (PointFrameResidual *r : ph->residuals) {
                            r->resetOOB();
                            r->linearize(&Hcalib);
                            r->efResidual->isLinearized = false;
                            r->applyRes(true);
                            if (r->efResidual->isActive()) {
                                r->efResidual->fixLinearizationF(ef);
                                ngoodRes++;
                            }
                        }
                        if (ph->idepth_hessian > setting_minIdepthH_marg) {
                            flag_inin++;
                            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
                            host->pointHessiansMarginalized.push_back(ph);
                        } else {
                            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                            host->pointHessiansOut.push_back(ph);
                        }

                    } else {
                        host->pointHessiansOut.push_back(ph);
                        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

                        // printf("drop point in frame %d (%d goodRes, %d activeRes)\n",
                        // ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
                    }

                    host->pointHessians[i] = 0;
                }
            }

            for (int i = 0; i < (int) host->pointHessians.size(); i++) {
                if (host->pointHessians[i] == 0) {
                    host->pointHessians[i] = host->pointHessians.back();
                    host->pointHessians.pop_back();
                    i--;
                }
            }
        }
    }

    void FullSystem::addActiveFrame(ImageAndExposure *image,
                                    ImageAndExposure *image_right, int id) {

        if (isLost)
            return;
        boost::unique_lock<boost::mutex> lock(trackMutex);

        // =========================== add into allFrameHistory
        // =========================
        FrameHessian *fh = new FrameHessian();
        FrameHessian *fh_right = new FrameHessian();
        FrameShell *shell = new FrameShell();
        shell->camToWorld =
                SE3(); // no lock required, as fh is not used anywhere yet.
        shell->aff_g2l = AffLight(0, 0);
        shell->marginalizedAt = shell->id = allFrameHistory.size();
        shell->timestamp = image->timestamp;
        shell->incoming_id = id; // id passed into DSO
        fh->shell = shell;
        fh_right->shell = shell;
        allFrameHistory.push_back(shell);

        // =========================== make Images / derivatives etc.
        // =========================
        fh->ab_exposure = image->exposure_time;
        fh->makeImages(image->image, &Hcalib);
        fh_right->ab_exposure = image_right->exposure_time;
        fh_right->makeImages(image_right->image, &Hcalib);

        if (!initialized) {
            // use initializer!
            if (coarseInitializer->frameID <
                0) // first frame set. fh is kept by coarseInitializer.
            {
                coarseInitializer->setFirstStereo(&Hcalib, fh, fh_right);
                initialized = true;
            }
            return;
        } else // do front-end operation.
        {
            // =========================== SWAP tracking reference?.
            // =========================
            if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
                boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
                CoarseTracker *tmp = coarseTracker;
                coarseTracker = coarseTracker_forNewKF;
                coarseTracker_forNewKF = tmp;
            }

            Vec4 tres = trackNewCoarse(fh, fh_right);
            if (!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1]) ||
                !std::isfinite((double) tres[2]) || !std::isfinite((double) tres[3])) {
                printf("Initial Tracking failed: LOST!\n");
                isLost = true;
                return;
            }

            bool needToMakeKF = false;

            if (setting_keyframesPerSecond > 0) {
                needToMakeKF =
                        allFrameHistory.size() == 1 ||
                        (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
                        0.95f / setting_keyframesPerSecond;
            } else {
                Vec2 refToFh = AffLight::fromToVecExposure(
                        coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                        coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

                float delta = setting_kfGlobalWeight * setting_maxShiftWeightT *
                              sqrtf((double) tres[1]) / (wG[0] + hG[0]) +
                              setting_kfGlobalWeight * setting_maxShiftWeightR *
                              sqrtf((double) tres[2]) / (wG[0] + hG[0]) +
                              setting_kfGlobalWeight * setting_maxShiftWeightRT *
                              sqrtf((double) tres[3]) / (wG[0] + hG[0]) +
                              setting_kfGlobalWeight * setting_maxAffineWeight *
                              fabs(logf((float) refToFh[0]));
                //            printf(" delta is %f \n", delta);
                // BRIGHTNESS CHECK
                needToMakeKF = allFrameHistory.size() == 1 || delta > 1 ||
                               2 * coarseTracker->firstCoarseRMSE < tres[0];
            }

            for (IOWrap::Output3DWrapper *ow : outputWrapper)
                ow->publishCamPose(fh->shell, &Hcalib);

            lock.unlock();
            deliverTrackedFrame(fh, fh_right, needToMakeKF);
            return;
        }
    }

    void FullSystem::deliverTrackedFrame(FrameHessian *fh, FrameHessian *fh_right,
                                         bool needKF) {

        if (linearizeOperation) {
            if (goStepByStep && lastRefStopID != coarseTracker->refFrameID) {
                MinimalImageF3 img(wG[0], hG[0], fh->dI);
                IOWrap::displayImage("frameToTrack", &img);
                while (true) {
                    char k = IOWrap::waitKey(0);
                    if (k == ' ')
                        break;
                    handleKey(k);
                }
                lastRefStopID = coarseTracker->refFrameID;
            } else
                handleKey(IOWrap::waitKey(1));

            if (needKF)
                makeKeyFrame(fh, fh_right);
            else
                makeNonKeyFrame(fh, fh_right);
        } else {
            boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
            unmappedTrackedFrames.push_back(fh);
            unmappedTrackedFrames_right.push_back(fh_right);
            if (needKF)
                needNewKFAfter = fh->shell->trackingRef->id;
            trackedFrameSignal.notify_all();

            while (coarseTracker_forNewKF->refFrameID == -1 &&
                   coarseTracker->refFrameID == -1) {
                mappedFrameSignal.wait(lock);
            }

            lock.unlock();
        }
    }

    void FullSystem::mappingLoop() {
        boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

        while (runMapping) {
            while (unmappedTrackedFrames.size() == 0) {
                trackedFrameSignal.wait(lock);
                if (!runMapping)
                    return;
            }

            FrameHessian *fh = unmappedTrackedFrames.front();
            unmappedTrackedFrames.pop_front();
            FrameHessian *fh_right = unmappedTrackedFrames_right.front();
            unmappedTrackedFrames_right.pop_front();

            // guaranteed to make a KF for the very first two tracked frames.
            if (allKeyFramesHistory.size() <= 2) {
                lock.unlock();
                makeKeyFrame(fh, fh_right);
                lock.lock();
                mappedFrameSignal.notify_all();
                continue;
            }

            //超过三帧没建地图，暂停Mapping
            if (unmappedTrackedFrames.size() > 3)
                needToKetchupMapping = true;

            //如果仍然有图片没建地图,则不把它设为keyframe了
            if (unmappedTrackedFrames.size() >
                0) // if there are other frames to track, do that first.
            {
                lock.unlock();
                makeNonKeyFrame(fh, fh_right);
                lock.lock();

                if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
                    FrameHessian *fh = unmappedTrackedFrames.front();
                    unmappedTrackedFrames.pop_front();
                    {
                        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                        assert(fh->shell->trackingRef != 0);
                        fh->shell->camToWorld =
                                fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
                        fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),
                                             fh->shell->aff_g2l);
                    }
                    delete fh;
                    delete fh_right;
                }

            } else {
                std::cout << " yes " << std::endl;
                if (setting_realTimeMaxKF ||
                    needNewKFAfter >= frameHessians.back()->shell->id) {
                    lock.unlock();
                    //关键操作！！插入关键帧！！
                    makeKeyFrame(fh, fh_right);
                    needToKetchupMapping = false;
                    lock.lock();
                } else {
                    lock.unlock();
                    makeNonKeyFrame(fh, fh_right);
                    lock.lock();
                }
            }
            mappedFrameSignal.notify_all();
        }
        printf("MAPPING FINISHED!\n");
    }

    void FullSystem::blockUntilMappingIsFinished() {
        boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
        runMapping = false;
        trackedFrameSignal.notify_all();
        lock.unlock();

        mappingThread.join();
    }

    void FullSystem::makeNonKeyFrame(FrameHessian *fh, FrameHessian *fh_right) {
        // needs to be set by mapping thread. no lock required since we are in mapping
        // thread.
        {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->camToWorld =
                    fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
        }

        traceNewCoarseNonKeyWithMatchPic(fh, fh_right);
        //    traceNewCoarseNonKey(fh, fh_right);
        delete fh;
        delete fh_right;
    }

    /**
     * 添加新的关键帧
     * 1. 跟踪当前帧的immature点，增加点的数量
     * 2. 标记需要边缘化的帧(主要标记那些帧中的点（PointHessian以及Immature Point）比较少的情况)
     * 3. 把当前帧插入待处理的帧队列。
     * 4. setPreCalc 这个现在还不能全部理解，只是知道一些先验相关的操作
     * 5. 看看active frame的immaturePoints能不能在当前帧被track到，并更新这些points的状态
     * 6. ef make index，赞不明白
     * 7. optimization
     * 8. 检验是否初始化成功
     * 9. remove outliers
     * 10. 标记需要移除和边缘化的点
     * 11. 移除点
     * 12. 获取零空间，FEJ
     * 13. 边缘化点
     * 14. 在生成keyframe时，要makeNewTrace,即更新参与trace的ImmaturePoints
     * 15. 进行可视化输出
     * 16. 边缘化帧
     * 17.
     * @param
     * @return
     */
    void FullSystem::makeKeyFrame(FrameHessian *fh, FrameHessian *fh_right) {
        // needs to be set by mapping thread
        {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            assert(fh->shell->trackingRef != 0);
            fh->shell->camToWorld =
                    fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
            fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
        }

        //	traceNewCoarseNonKeyWithMatchPic(fh, fh_right);
        traceNewCoarse(fh, fh_right);

        boost::unique_lock<boost::mutex> lock(mapMutex);

        // =========================== Flag Frames to be Marginalized.
        // =========================
        flagFramesForMarginalization(fh);

        // =========================== add New Frame to Hessian Struct.
        // =========================
        fh->idx = frameHessians.size();
        frameHessians.push_back(fh);

        fh->frameID = allKeyFramesHistory.size();
        allKeyFramesHistory.push_back(fh->shell);
        ef->insertFrame(fh, &Hcalib);

        setPrecalcValues();

        // =========================== add new residuals for old points
        // =========================
        int numFwdResAdde = 0;
        for (FrameHessian *fh1 : frameHessians) // go through all active frames
        {
            if (fh1 == fh)
                continue;
            for (PointHessian *ph : fh1->pointHessians) {
                PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh);
                r->setState(ResState::IN);
                ph->residuals.push_back(r);
                ef->insertResidual(r);
                ph->lastResiduals[1] = ph->lastResiduals[0];
                ph->lastResiduals[0] =
                        std::pair<PointFrameResidual *, ResState>(r, ResState::IN);
                numFwdResAdde += 1;
            }
        }

        // =========================== Activate Points (& flag for marginalization).
        // =========================
        activatePointsMT();
        ef->makeIDX();

        // =========================== OPTIMIZE ALL =========================
        fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
        float rmse = optimize(setting_maxOptIterations);
        // printf("allKeyFramesHistory size is %d \n",
        // (int)allKeyFramesHistory.size());
        printf("rmse is %f \n", rmse);

        // =========================== Figure Out if INITIALIZATION FAILED
        // ========================= 检查初始化的结果，那几个阈值怎么搞来的。。。。
        if (allKeyFramesHistory.size() <= 4) {
            if (allKeyFramesHistory.size() == 2 &&
                rmse > 20 * benchmark_initializerSlackFactor) {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
            if (allKeyFramesHistory.size() == 3 &&
                rmse > 13 * benchmark_initializerSlackFactor) {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
            if (allKeyFramesHistory.size() == 4 &&
                rmse > 9 * benchmark_initializerSlackFactor) {
                printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
                initFailed = true;
            }
        }
        //只要Lost了就停止return，下面的不执行。
        if (isLost)
            return;

        // =========================== REMOVE OUTLIER =========================
        removeOutliers();

        {
            boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
            coarseTracker_forNewKF->makeK(&Hcalib);
            coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, fh_right,
                                                         Hcalib);

            coarseTracker_forNewKF->debugPlotIDepthMap(
                    &minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
            coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
        }

        //	debugPlot("post Optimize");

        // =========================== (Activate-)Marginalize Points
        // =========================
        flagPointsForRemoval();
        ef->dropPointsF();
        getNullspaces(ef->lastNullspaces_pose, ef->lastNullspaces_scale,
                      ef->lastNullspaces_affA, ef->lastNullspaces_affB);
        ef->marginalizePointsF();

        // =========================== add new Immature points & new residuals
        // =========================
        //对于加进去的最新关键帧，在这一步才开始选immature point!!
        makeNewTraces(fh, fh_right, 0);

        /*	for(ImmaturePoint* ph : fh->immaturePoints)
                {
                        ph->u_stereo = ph->u;
                        ph->v_stereo = ph->v;
                        ph->idepth_min_stereo = 0;
                        ph->idepth_max_stereo = NAN;

                        ImmaturePointStatus phStatus = ph->traceright(fh_right, K);
                        if(phStatus == ImmaturePointStatus::IPS_GOOD)
                        {
                                ph->idepth_min = ph->idepth_min_stereo;
                                ph->idepth_max = ph->idepth_max_stereo;
                        }

                }*/

        for (IOWrap::Output3DWrapper *ow : outputWrapper) {
            ow->publishGraph(ef->connectivityMap);
            ow->publishKeyframes(frameHessians, false, &Hcalib);
        }

        // =========================== Marginalize Frames =========================

        for (unsigned int i = 0; i < frameHessians.size(); i++) {
            if (frameHessians[i]->flaggedForMarginalization) {
                marginalizeFrame(frameHessians[i]);
                i = 0;
            }
        }

        delete fh_right;

        //	printLogLine();
        //    printEigenValLine();
    }

//将firstFrame作为关键帧插入到FrameHessians结构中
    void FullSystem::initializeFromInitializer(FrameHessian *newFrame) {
        boost::unique_lock<boost::mutex> lock(mapMutex);

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib.fxl();
        K(1, 1) = Hcalib.fyl();
        K(0, 2) = Hcalib.cxl();
        K(1, 2) = Hcalib.cyl();
        float bf = K(0, 0) * 0.53716572;

        // add firstframe.
        FrameHessian *firstFrame = coarseInitializer->firstFrame;
        firstFrame->idx = frameHessians.size();
        frameHessians.push_back(firstFrame);
        firstFrame->frameID = allKeyFramesHistory.size();
        allKeyFramesHistory.push_back(firstFrame->shell);
        ef->insertFrame(firstFrame, &Hcalib);
        setPrecalcValues();

        FrameHessian *firstFrameRight = coarseInitializer->firstRightFrame;
        frameHessiansRight.push_back(firstFrameRight);

        firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
        firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
        firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

        float idepthStereo = 0;
        float sumID = 1e-5, numID = 1e-5;
        for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
            sumID += coarseInitializer->points[0][i].iR;
            numID++;
        }
        float rescaleFactor = 1 / (sumID / numID);

        // randomly sub-select the points I need.
        float keepPercentage =
                setting_desiredPointDensity / coarseInitializer->numPoints[0];

        if (!setting_debugout_runquiet)
            printf("Initialization: keep %.1f%% (need %d, have %d)!\n",
                   100 * keepPercentage, (int) (setting_desiredPointDensity),
                   coarseInitializer->numPoints[0]);

        //对于Initializer选出来的每一个点
        for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
            if (rand() / (float) RAND_MAX > keepPercentage)
                continue;

            Pnt *point = coarseInitializer->points[0] + i;
            ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f,
                                                  firstFrame, point->my_type, &Hcalib);

            //        float sgmDisparity =
            //        getMatInterpolatedElement11BiLin(firstDisparity, pt->u, pt->v);

            pt->u_stereo = pt->u;
            pt->v_stereo = pt->v;
            pt->idepth_min_stereo = 0;
            pt->idepth_max_stereo = NAN;
            pt->traceright(firstFrameRight, K);
            pt->idepth_min = pt->idepth_min_stereo;
            pt->idepth_max = pt->idepth_max_stereo;
            idepthStereo = pt->idepth_stereo;

            //        pt->idepth_min = 0.95 * (sgmDisparity / bf);
            //        pt->idepth_max = 1.05 * (sgmDisparity / bf);
            //        idepthStereo = sgmDisparity / bf;

            if (!std::isfinite(pt->energyTH) || !std::isfinite(pt->idepth_min) ||
                !std::isfinite(pt->idepth_max) || pt->idepth_min < 0 ||
                pt->idepth_max < 0) {
                delete pt;
                continue;
            }

            PointHessian *ph = new PointHessian(pt, &Hcalib);
            delete pt;
            if (!std::isfinite(ph->energyTH)) {
                delete ph;
                continue;
            }

            ph->setIdepthScaled(idepthStereo);
            ph->setIdepthZero(idepthStereo);
            ph->hasDepthPrior = true;
            ph->setPointStatus(PointHessian::ACTIVE);

            firstFrame->pointHessians.push_back(ph);
            ef->insertPoint(ph);
        }

        //    printf("we got %d point \n", (int)firstFrame->pointHessians.size());

        SE3 firstToNew = coarseInitializer->thisToNext;
        //	firstToNew.translation() /= rescaleFactor;

        // really no lock required, as we are initializing.
        {
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            firstFrame->shell->camToWorld = SE3();
            firstFrame->shell->aff_g2l = AffLight(0, 0);
            firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),
                                         firstFrame->shell->aff_g2l);
            firstFrame->shell->trackingRef = 0;
            firstFrame->shell->camToTrackingRef = SE3();

            newFrame->shell->camToWorld = firstToNew.inverse();
            newFrame->shell->aff_g2l = AffLight(0, 0);
            newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),
                                       newFrame->shell->aff_g2l);
            newFrame->shell->trackingRef = firstFrame->shell;
            newFrame->shell->camToTrackingRef = firstToNew.inverse();
        }

        initialized = true;
        printf("INITIALIZE FROM INITIALIZER (%d pts)!\n",
               (int) firstFrame->pointHessians.size());
        // printf("the size of framehessian is %d , %d, %d, %d \n",
        // frameHessians.size(), allKeyFramesHistory.size(), firstFrame->frameID,
        // newFrame->frameID);
    }

//在生成keyframe时，要makeNewTrace,即更新参与trace的ImmaturePoints
    void FullSystem::makeNewTraces(FrameHessian *newFrame,
                                   FrameHessian *newFrameRight, float *gtDepth) {
        pixelSelector->allowFast = true;
        // int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0],
        // hG[0], setting_desiredDensity);
        int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,
                                                     setting_desiredImmatureDensity);

        newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
        // fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
        newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
        newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

        Mat33f K = Mat33f::Identity();
        K(0, 0) = Hcalib.fxl();
        K(1, 1) = Hcalib.fyl();
        K(0, 2) = Hcalib.cxl();
        K(1, 2) = Hcalib.cyl();
        int delete_num = 0;

        for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
            for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++) {
                //第y行x列像素
                int i = x + y * wG[0];
                if (selectionMap[i] == 0) {
                    delete_num++;
                    continue;
                }

                ImmaturePoint *impt =
                        new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);

                if (!std::isfinite(impt->energyTH)) {
                    delete impt;
                } else
                    newFrame->immaturePoints.push_back(impt);
            }
        printf("MADE %d IMMATURE POINTS! Delete %d \n", (int) newFrame->immaturePoints.size(), delete_num);
    }
    /*
     * pre calc
    * @param
    * @return
    */
    void FullSystem::setPrecalcValues() {
        for (FrameHessian *fh : frameHessians) {
            fh->targetPrecalc.resize(frameHessians.size());
            for (unsigned int i = 0; i < frameHessians.size(); i++)
                fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
        }

        ef->setDeltaF(&Hcalib);
    }

    void FullSystem::printLogLine() {
        if (frameHessians.size() == 0)
            return;

        if (!setting_debugout_runquiet)
            printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. "
                           "a=%f, b=%f. Window %d (%d)\n",
                   allKeyFramesHistory.back()->id, statistics_lastFineTrackRMSE,
                   ef->resInA, ef->resInL, ef->resInM,
                   (int) statistics_numForceDroppedResFwd,
                   (int) statistics_numForceDroppedResBwd,
                   allKeyFramesHistory.back()->aff_g2l.a,
                   allKeyFramesHistory.back()->aff_g2l.b,
                   frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                   (int) frameHessians.size());

        if (!setting_logStuff)
            return;

        if (numsLog != 0) {
            (*numsLog) << allKeyFramesHistory.back()->id << " "
                       << statistics_lastFineTrackRMSE << " "
                       << (int) statistics_numCreatedPoints << " "
                       << (int) statistics_numActivatedPoints << " "
                       << (int) statistics_numDroppedPoints << " "
                       << (int) statistics_lastNumOptIts << " " << ef->resInA << " "
                       << ef->resInL << " " << ef->resInM << " "
                       << statistics_numMargResFwd << " " << statistics_numMargResBwd
                       << " " << statistics_numForceDroppedResFwd << " "
                       << statistics_numForceDroppedResBwd << " "
                       << frameHessians.back()->aff_g2l().a << " "
                       << frameHessians.back()->aff_g2l().b << " "
                       << frameHessians.back()->shell->id -
                          frameHessians.front()->shell->id
                       << " " << (int) frameHessians.size() << " "
                       << "\n";
            numsLog->flush();
        }
    }

    void FullSystem::printEigenValLine() {
        if (!setting_logStuff)
            return;
        if (ef->lastHS.rows() < 12)
            return;

        MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS,
                                                ef->lastHS.cols() - CPARS);
        MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS,
                                                ef->lastHS.cols() - CPARS);
        int n = Hp.cols() / 8;
        assert(Hp.cols() % 8 == 0);

        // sub-select
        for (int i = 0; i < n; i++) {
            MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
            Hp.block(i * 6, 0, 6, n * 8) = tmp6;

            MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
            Ha.block(i * 2, 0, 2, n * 8) = tmp2;
        }
        for (int i = 0; i < n; i++) {
            MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
            Hp.block(0, i * 6, n * 8, 6) = tmp6;

            MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
            Ha.block(0, i * 2, n * 8, 2) = tmp2;
        }

        VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
        VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
        VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
        VecX diagonal = ef->lastHS.diagonal();

        std::sort(eigenvaluesAll.data(),
                  eigenvaluesAll.data() + eigenvaluesAll.size());
        std::sort(eigenP.data(), eigenP.data() + eigenP.size());
        std::sort(eigenA.data(), eigenA.data() + eigenA.size());

        int nz = std::max(100, setting_maxFrames * 10);

        if (eigenAllLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
            (*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                           << "\n";
            eigenAllLog->flush();
        }
        if (eigenALog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(eigenA.size()) = eigenA;
            (*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                         << "\n";
            eigenALog->flush();
        }
        if (eigenPLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(eigenP.size()) = eigenP;
            (*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                         << "\n";
            eigenPLog->flush();
        }

        if (DiagonalLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(diagonal.size()) = diagonal;
            (*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                           << "\n";
            DiagonalLog->flush();
        }

        if (variancesLog != 0) {
            VecX ea = VecX::Zero(nz);
            ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
            (*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose()
                            << "\n";
            variancesLog->flush();
        }

        std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
        (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
        for (unsigned int i = 0; i < nsp.size(); i++)
            (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " "
                             << nsp[i].dot(ef->lastbS) << " ";
        (*nullspacesLog) << "\n";
        nullspacesLog->flush();
    }

    void FullSystem::printFrameLifetimes() {
        if (!setting_logStuff)
            return;

        boost::unique_lock<boost::mutex> lock(trackMutex);

        std::ofstream *lg = new std::ofstream();
        lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
        lg->precision(15);

        for (FrameShell *s : allFrameHistory) {
            (*lg) << s->id << " " << s->marginalizedAt << " "
                  << s->statistics_goodResOnThis << " "
                  << s->statistics_outlierResOnThis << " " << s->movedByOpt;

            (*lg) << "\n";
        }

        lg->close();
        delete lg;
    }

    void FullSystem::printEvalLine() { return; }

} // namespace dso
