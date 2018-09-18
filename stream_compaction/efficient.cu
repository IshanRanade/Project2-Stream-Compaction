#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <device_launch_parameters.h>


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpsweep(int n, int power, int *array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                for (int k = 0; k < n; k += 2 * power) {
                    array[k + (2 * power) - 1] = array[k + power - 1] + array[k + (2 * power) - 1];
                }
            }
        }

        __global__ void kernDownsweep(int n, int power, int *array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                for (int k = 0; k < n; k += 2 * power) {
                    int t = array[k + power - 1];
                    array[k + power - 1] = array[k + (2 * power) - 1];
                    array[k + (2 * power) - 1] = t + array[k + (2 * power) - 1];
                }
            }
        }

        __global__ void kernSetZero(int n, int *array) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                if (index == n - 1) {
                    array[index] = 0;
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *temp;
            cudaMalloc((void**)&temp, n * sizeof(int));
            cudaMemcpy(temp, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            cudaDeviceSynchronize();

            timer().startGpuTimer();

            int blockSize = 256;
            int blocks = (n + blockSize - 1) / blockSize;

            // TODO
            for (int d = 0; d < ilog2ceil(n); ++d) {
                kernUpsweep << <blocks, blockSize >> > (n, pow(2, d), temp);
                cudaDeviceSynchronize();
            }
            
            kernSetZero << <blocks, blockSize >> > (n, temp);

            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                kernDownsweep << <blocks, blockSize >> > (n, pow(2, d), temp);
                cudaDeviceSynchronize();
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, temp, n * sizeof(int), cudaMemcpyDeviceToHost);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            
            // TODO

            timer().endGpuTimer();
            return -1;
        }
    }
}
