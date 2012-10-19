/*
 * Title: martrixMut.cu
 * Author: 陈志韬
 * Student ID: SA12011089
 */
#include <stdio.h>
#include<stdlib.h>
#include<time.h>

#define MAX_NUM_SIZE 100
#define DEFAULT_M 400
#define DEFAULT_N 400
#define DEFAULT_O 400
#define DEFAULT_BLOCK_SIZE 512


__global__ void kernel(int m, int n, int o, int *a, int *b, int *c)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx > m * o)
    {
        return;
    }
    int i = idx / o;
    int j = idx % o;
    c[i * o + j] = 0;
    for(int k = 0; k < n; k++)
    {
        c[i * o + j] += a[i * n + k] * b[k * o + j];
    }
}


void randomInit(int *array, int m)
{
    srand(time(0));
    for(int i = 0;i < m; i++)
    {
        array[i] = random() % MAX_NUM_SIZE;
    }
}

void cpuCalc( int m, int n, int o, int *a, int *b, int *c)
{
    for(int i = 0;i < m;i++)
        for(int j = 0;j < o; j++)
        {
            c[i * o + j] = 0;
            for(int k = 0; k < n; k++)
            {
                c[i * o + j] += a[i * n + k] * b[k * o + j];
            }
        }
}
void printMatrix(int *array, int m, int n)
{
    for(int i=0; i<m*n; i++)
    {
        if(0 != i && 0 == i % n)
            printf("\n");
        printf("%d ", array[i] );
    }
    printf("\n");
}

void compareMatrix(int *a, int *b, int m, int n)
{
    bool isSame = true;
    for(int i = 0;i < m;i++)
        for(int j = 0;j < n; j++)
        {
            if( a[i*n +j] != b[i*n + j])
            {
                printf("array[%d][%d] is different, with %d and %d\n", i, j, a[i*m+j], b[i*m+j]);
                isSame = false;
            }
        }
    if(isSame)
        printf("The result is same!!\n");
    else
        printf("The resutl is Different!!\n");
}

int main(int argc, char* argv[])
{
    int m = DEFAULT_M, n = DEFAULT_N, o = DEFAULT_O;
    if(argc == 1)
    {
        printf("Usage: %s 200 300 400 256\n", argv[0]);
        printf("Stand for C = A*B, A是200*300的矩阵，B是300*400的矩阵，256指的是blockDim大小\n");
        printf("默认是 400 400 400 512, blockDim最大允许512，否则会在较大的矩阵相乘时出现错误。\n");
        printf("可以只输入前面任何几个参数，均可以识别。\n");
        printf("最大测试数据为：a.out 2000 2000 2000 512，输出结果表示正确\n");
    }
    if(argc > 1)
    {
        m = atoi(argv[1]);
    }
    if(argc > 2)
    {
        n = atoi(argv[2]);
    }
    if(argc > 3)
    {
        o = atoi(argv[3]);
    }
    int num_a = m*n;
    int num_b = n*o;
    int num_c = m*o;
    int num_bytes_a = num_a * sizeof(int);
    int num_bytes_b = num_b * sizeof(int);
    int num_bytes_c = num_c * sizeof(int);
    int *d_a = 0, *h_a = 0; 
    int *d_b = 0, *h_b = 0; 
    int *d_c = 0, *h_c = 0; 
    int *cpu = 0;
    h_a = (int*)malloc(num_bytes_a);
    h_b = (int*)malloc(num_bytes_b);
    h_c = (int*)malloc(num_bytes_c);
    cpu = (int*)malloc(num_bytes_c);
    cudaMalloc( (void**)&d_a, num_bytes_a );
    cudaMalloc( (void**)&d_b, num_bytes_b );
    cudaMalloc( (void**)&d_c, num_bytes_c );
    if( 0==h_a || 0==d_a || 0==h_b || 0==d_b || 0==h_c || 0==d_c )
    {
        printf("couldn't allocate memory\n");
        return 1;
    }
    //初始化，并且把初始化后的数据传入GPU
    randomInit(h_a, num_a);
    randomInit(h_b, num_b);
    cudaMemcpy( d_a, h_a, num_bytes_a, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, h_b, num_bytes_b, cudaMemcpyHostToDevice );
    cudaMemset( d_c, 0, num_bytes_c );

    //开始进入GPU执行
    dim3 grid, block;
    block.x = DEFAULT_BLOCK_SIZE;
    if(argc > 4)
    {
        block.x = atoi(argv[4]);
    }
    grid.x = (num_c + block.x - 1)/block.x;
    cudaEvent_t start, stop;
    cudaEventCreate(&start),
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    kernel<<<grid, block>>>(m, n, o, d_a, d_b, d_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time; 
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Time elapsed: %fms\n", time);

    cudaMemcpy( h_c, d_c, num_bytes_c, cudaMemcpyDeviceToHost );
    clock_t start_cpu = clock();
    cpuCalc(m, n, o, h_a, h_b, cpu);
    clock_t end_cpu = clock();
    float cpu_time = ((double)end_cpu - start_cpu)*1000/CLOCKS_PER_SEC;
    printf("Cpu Time Elapsed: %fms\n", cpu_time);
    /*printf("A:\n");*/
    /*printMatrix(h_a, m, n);*/
    /*printf("B:\n");*/
    /*printMatrix(h_b, n, o);*/
    /*printf("C=A*B:\n");*/
    /*printMatrix(h_c, m, o);*/
    /*printf("Cpu calc:\n");*/
    /*printMatrix(cpu , m, o);*/
    compareMatrix(h_c, cpu, m, o);

    free( h_a );
    free( h_b );
    free( h_c );
    free( cpu );
    cudaFree( d_a );
    cudaFree( d_b );
    cudaFree( d_c );
    return 0;
}
