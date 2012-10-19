#include <stdio.h>
__global__ void kernel(int *a)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    a[idx] = idx;
}
int main()
{
    int dimx = 16;
    int num_bytes = dimx*sizeof(int);
    int *d_a=0, *h_a=0; 
    int *s_a = 0;
    h_a = (int*)malloc(num_bytes);
    s_a = (int*)malloc(num_bytes);
    cudaMalloc( (void**)&d_a, num_bytes );
    if( 0==h_a || 0==d_a )
    {
        printf("couldn't allocate memory\n");
        return 1;
    }
    for(int i=0; i<dimx; i++)
    {
        s_a[i] = i;
        h_a[i] = i;
    }
    cudaMemset( d_a, 0, num_bytes );
    cudaMemcpy( d_a, s_a, num_bytes, cudaMemcpyHostToDevice );
    dim3 grid, block;
    block.x = 4;
    grid.x = dimx/block.x;
    /*kernel<<<grid, block>>>(d_a);*/
    cudaMemcpy( h_a, d_a, num_bytes, cudaMemcpyDeviceToHost );
    for(int i=0; i<dimx; i++)
        printf("%d ", h_a[i] );
    printf("\n");
    free( h_a );
    cudaFree( d_a );
    return 0;
}
