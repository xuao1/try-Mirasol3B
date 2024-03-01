#include <stdio.h>
#include <cuda_runtime.h>

#define N 500000 // 数组大小，使得内核运行时间为几微秒
#define NSTEP 100 // 循环步数
#define NKERNEL 20 // 内核次数

// CUDA 内核函数定义
__global__ void shortKernel(float *out_d, float *in_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out_d[idx] = 1.23 * in_d[idx];
}

int main() {
    printf("Begin...\n");
    float *in_d, *out_d; // 设备端指针
    float *in_h, *out_h; // 主机端指针
    int size = N * sizeof(float); // 分配的内存大小

    // 分配主机端内存
    in_h = (float *)malloc(size);
    out_h = (float *)malloc(size);

    // 初始化输入数组
    for (int i = 0; i < N; i++) {
        in_h[i] = i;
    }

    // 分配设备端内存
    cudaMalloc((void **)&in_d, size);
    cudaMalloc((void **)&out_d, size);

    // 主机到设备的数据传输
    cudaMemcpy(in_d, in_h, size, cudaMemcpyHostToDevice);

    // 配置内核执行参数
    int threads = 512;
    int blocks = (N + threads - 1) / threads;

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动 CPU 计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    printf("Exec...\n");

    // 内核执行循环
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    for(int istep = 0; istep < NSTEP; istep++){
        if(!graphCreated){
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for(int ikrnl = 0; ikrnl < NKERNEL; ikrnl++){
                shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated=true;
        }
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
    }

    printf("Done\n");

    // 结束 CPU 计时器
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Elapsed time: %f ms\n", milliseconds);

    // 设备到主机的数据传输
    cudaMemcpy(out_h, out_d, size, cudaMemcpyDeviceToHost);

    // 释放设备端内存
    cudaFree(in_d);
    cudaFree(out_d);

    // 释放主机端内存
    free(in_h);
    free(out_h);

    // 销毁 CUDA 流
    cudaStreamDestroy(stream);

    return 0;
}
