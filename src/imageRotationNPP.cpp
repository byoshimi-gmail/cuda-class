/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char *filePath;
        float angleF = 45.0;

        findCudaDevice(argc, (const char **)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "angle"))
        {
            angleF = getCmdLineArgumentFloat(argc, (const char **)argv, "angle");
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "data/Lena-grey.pgm";
        }

        // if we specify the filename at the command line, then we only test
        // sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "nppiRotate opened: <" << sFilename.data()
                      << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "nppiRotate unable to open: <" << sFilename.data() << ">"
                      << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_rotate.pgm";

        if (checkCmdLineFlag(argc, (const char **)argv, "output"))
        {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output",
                                     &outputFilePath);
            sResultFilename = outputFilePath;
        }

        // NOTE: this code only works for 8-bit grey scale.  npp::loadImage() defined in
        // Common/UtilNPP/ImageIO.h   It only support 8-bit grey decoding.

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc; // works for 8 bit grayscale
        npp::loadImage(sFilename, oHostSrc);
        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        // create struct with the ROI size
        NppiRect oSrcSize = {0, 0, (int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiPoint oSrcOffset = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height() };
        NppiRect oSizeROI = {0, 0, (int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        // Calculate the bounding box of the rotated image

        double angle = (double) angleF;
        double aBoundingBox[2][2] = {
            {0, 0},
            {(double)oDeviceSrc.width(), (double)oDeviceSrc.height()}};

        double oBoundingBox[2][2];
        double oRotatedBoundingBox[2][2];

        // Maximal bounding box, at least for square case is 45 deg.
        // Better check would use sqrt(width^2 + height^2) as output width and height.
        NPP_CHECK_NPP(nppiGetRotateBound(oSrcSize, oBoundingBox, 45.0/*angle*/, 0, 0));
        NPP_CHECK_NPP(nppiGetRotateBound(oSrcSize, oRotatedBoundingBox, angle, 0, 0));

        // allocate device image for the rotated image
        npp::ImageNPP_8u_C1 oDeviceDst(oBoundingBox[1][0] - oBoundingBox[0][0],
                                       oBoundingBox[1][1] - oBoundingBox[0][1]);

        NppiRect oBoundingRect = {0, 0,
                                  (int)oBoundingBox[1][0] - (int)oBoundingBox[0][0],
                                  (int)oBoundingBox[1][1] - (int)oBoundingBox[0][1]};

        NppiSize oSrcSizeSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        NppiSize oSrcOffsetSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        // run the rotation
        //for (angle = 5; angle < 360; angle+=5) {
        std::cout << "angle = " << angle << "\n";
        std::cout << "output image x, y offsets = " << -(int)oBoundingBox[0][0] << ", " << -(int)oBoundingBox[0][1] << " - "
                  << -(int)oBoundingBox[1][0] << ", " << -(int)oBoundingBox[1][1] << "\n";
        std::cout << "rotated x, y offsets = " << -(int)oRotatedBoundingBox[0][0] << ", " << -(int)oRotatedBoundingBox[0][1] << " - "
                  << -(int)oRotatedBoundingBox[1][0] << ", " << -(int)oRotatedBoundingBox[1][1] << "\n";
        NPP_CHECK_NPP(nppiRotate_8u_C1R(
            oDeviceSrc.data(), oSrcSizeSize, oDeviceSrc.pitch(), oSrcSize,
            oDeviceDst.data(), oDeviceDst.pitch(), oBoundingRect, angle,
            -(int)oRotatedBoundingBox[0][0], -(int)oRotatedBoundingBox[0][1],
            NPPI_INTER_NN));
        //}

        // declare a host image for the result
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        // and copy the device result data into it
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        saveImage(sResultFilename, oHostDst);
        std::cout << "Saved image: " << sResultFilename << std::endl;

        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
