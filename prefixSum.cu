/*
 * Title: prefixScan.cu
 * Author: 陈志韬
 * Student ID: SA12011089
 */
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
/*#include<c*/

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 

#define DATA_SIZE 32
#define DEFAULT_BLOCK_SIZE 256
#define RAND_INT 50
//Only generate integers with value 0 or 1
#define MAX_INT_SIZE 2
typedef float calcType;

void randomInit(calcType *array, int m)
{
    /*srand(time(0));*/
    for(int i = 0;i < m; i++)
    {
        array[i] = ((calcType)random() / RAND_MAX) * RAND_INT;
    }
}

void randomInit(int *array, int m)
{
    /*srand(time(0));*/
    for(int i = 0;i < m; i++)
    {
        array[i] = random()%MAX_INT_SIZE;
    }
}

void printArray(calcType * array, int m)
{
    int i = 0;
    for(i = 0;i < m; i++)
    {
        printf("%f ", array[i]);
    }
    printf("\n");
}

//The size of out is larger than in by one, i.e the size of m is m+1
void cpuCalc(calcType *out, const calcType *in, int m)
{

    int j = 0;
    out[0] = 0;
    for(j = 0;j < m;j++)
    {
        out[j] = out[j - 1] + in[j - 1];
    }
}

__global__ void prescanEasy(calcType *g_odata, calcType *g_idata, int n)
{
    extern __shared__ calcType temp[];
    int thid = threadIdx.x;
    int pout = 0, pin = 1;

    //Make sure the data has been transfomed into the kernel thread
    /*g_odata[thid] = g_idata[thid];*/
    temp[pout*n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
    //下面一行GPU Gems 3没有，是我在进行错误的测试的时候认为是错误点添加上的
    //但是在修改73行代码之后的时候测试，不对结果产生影响。。
    //temp[pin*n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
    __syncthreads();
    /*for(int offset = 1; offset < n; offset *= 2)*/
    for(int offset = 1; offset < n; offset *= 2)
    {
        pout = 1 - pout; //交换双缓冲区的索引
        pin = 1 - pin;
        if( thid >= offset)
        {
            //GPU Gems 3 的样例程序的代码如下：
            //temp[pout*n+thid] += temp[pin*n+thid - offset];
            temp[pout*n + thid] = temp[pin*n + thid] + temp[pin*n + thid -offset];
        }
        else
            temp[pout*n + thid] = temp[pin*n + thid];
        __syncthreads();
        
    }
    g_odata[thid] = temp[pout*n + thid];

}
__global__ void prescanBank(calcType *g_odata, calcType *g_idata, int n)
{
    extern __shared__ calcType temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    int d, ai, bi;
    calcType t;
    
    //Below is A
    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];

    //A end
    for (d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            //Below is B
            ai = offset*(2*thid+1)-1;
            bi = offset*(2*thid+2)-1;
            //B end
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    //Below is C
    if (thid == 0) { temp[n - 1] = 0; } // clear the last element
    //C end
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            //Below is D
            ai = offset*(2*thid+1)-1;
            bi = offset*(2*thid+2)-1;
            //D end
            t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    //Below is E
    g_odata[2*thid] = temp[2*thid]; // write results to device memory
    g_odata[2*thid+1] = temp[2*thid+1];
    //E end

}

__global__ void prescan(calcType *g_odata, calcType *g_idata, int n)
{
    extern __shared__ calcType temp[];  // allocated on invocation
    int thid = threadIdx.x;
    int offset = 1;
    //Below is A
    /*temp[2*thid] = g_idata[2*thid]; // load input into shared memory*/
    /*temp[2*thid+1] = g_idata[2*thid+1];*/
    int ai = thid;
    int bi = thid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    //A end
    for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
    {
        __syncthreads();
        if (thid < d)
        {
            //Below is B
            /*int ai = offset*(2*thid+1)-1;*/
            /*int bi = offset*(2*thid+2)-1;*/
            int ai = offset * (2*thid + 1) - 1;
            int bi = offset * (2*thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            //B end
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    //Below is C
    /*if (thid == 0) { temp[n - 1] = 0; } // clear the last element*/
    if(thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n -1)] = 0;}
    //C end
    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            //Below is D
            /*int ai = offset*(2*thid+1)-1;*/
            /*int bi = offset*(2*thid+2)-1;*/
            int ai = offset * (2*thid + 1) - 1;
            int bi = offset * (2*thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            //D end
            calcType t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    //Below is E
    /*g_odata[2*thid] = temp[2*thid]; // write results to device memory*/
    /*g_odata[2*thid+1] = temp[2*thid+1];*/
    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
    //E end
}

int main(int argc, char *argv[])
{
    int data_bytes = DATA_SIZE * sizeof(calcType);
    
    calcType *h_out = 0, *d_out = 0;
    calcType *h_in  = 0, *d_in = 0;
    calcType *cpu = 0;
    h_out = (calcType*)malloc(data_bytes);
    h_in = (calcType*)malloc(data_bytes);
    cpu = (calcType*)malloc(data_bytes);
    cudaMalloc( (void**)&d_out, data_bytes);
    cudaMalloc( (void**)&d_in, data_bytes);
    if(0 == h_out || 0 == h_in || 0 == d_out || 0 == d_in)
    {
        printf("Couldn't allocate memory\n");
        return 1;
    }
    randomInit(h_in, DATA_SIZE);
    printArray(h_in, DATA_SIZE);
    cudaMemcpy(d_in, h_in, data_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, data_bytes);

    //Cpu Calc
    cpuCalc(cpu, h_in, DATA_SIZE);
    printArray(cpu, DATA_SIZE);

    //开始进入GPU执行
    dim3 grid, block;
    float time; 
    block.x = DATA_SIZE;
    /*grid.x = (DATA_SIZE + block.x - 1)/block.x;*/
    grid.x = 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start),
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    prescanEasy<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);
    /*prescanBank<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);*/
    /*prescan<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("PrescanEasy Time elapsed: %fms\n", time);

    cudaEventRecord(start, 0);
    /*prescanEasy<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);*/
    prescanBank<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);
    /*prescan<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("PrescanBank Time elapsed: %fms\n", time);

    cudaEventRecord(start, 0);
    /*prescanEasy<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);*/
    /*prescanBank<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);*/
    prescan<<<grid, block, data_bytes * 2>>>(d_out, d_in, DATA_SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Prescan Time elapsed: %fms\n", time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy( h_out, d_out, data_bytes, cudaMemcpyDeviceToHost );
    printArray(h_out, DATA_SIZE);
    free(h_in);
    free(h_out);
    free(cpu);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
