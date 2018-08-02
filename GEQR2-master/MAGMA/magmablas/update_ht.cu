#include "magma_internal.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

//==============================================================================
//==============================================================================
__global__ 
void stemp(float *matrix, float *dot, float *temp)
{
    float beta = sqrt(*dot);
    *temp = -copysign(beta, *matrix);
}

__global__ 
void scnst(int N, float *matrix, int ldda, float *dot, float *temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;	

    __syncthreads();

    if(i<N)
    {
	dot[i] = MAGMA_S_DIV(dot[i], temp[0]*(matrix[0]-temp[0])) - MAGMA_S_DIV(matrix[ldda*i], (matrix[0]-temp[0]));
    }

} 

__global__ 
void sdiff(float *matrix, float *diff, float *temp)
{
    *diff = (*matrix - *temp);
}

__global__
void srow1(int N, float *matrix, int ldda, float *dot, float *diff) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;		
 
    float ltemp;
 
    __syncthreads();	

    if(i<N)		
    {
	ltemp = matrix[ldda*i] + MAGMA_S_MUL(dot[i], *diff);
	__syncthreads();
	matrix[ldda*i] = ltemp;
    }

}

__global__
void stmup(int M, float *matrix, int ldda, float *dot, float *vector) 
{
    const int tx = threadIdx.x;

    dot    = dot + blockIdx.x;
    matrix = matrix + blockIdx.x * ldda;

    float tmp;

    if(blockIdx.x != 0){
    	tmp = dot[0];
    	for( int j = M-tx-1; j > 0; j -= BLOCK_SIZE )
	    matrix[j] += tmp*vector[j];
    }
}

__global__
void htcns(int M, float *vector, float *dtau, float *diff) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;		
  
    __syncthreads();	

    if(i == 0)
	*dtau = -(*diff)/(*vector);

    __syncthreads();

    if(i>0 && i<M)		
    {
	vector[i] = vector[i]/(*diff);
    }

}

extern "C" magma_int_t
update_HT(
    cublasHandle_t handle,
    magma_int_t m,  magma_int_t n,
    magmaFloat_ptr dv,
    magmaFloat_ptr dtau,
    magmaFloat_ptr dC,  magma_int_t lddc,
    magmaFloat_ptr dwork,
    magma_queue_t queue )
{
    #define dC(i_,j_) (dC + (i_) + (j_)*(lddc))

    float alpha, beta;

    alpha  = MAGMA_S_ONE;
    beta   = MAGMA_S_ZERO;

    if( (m % BLOCK_SIZE) != 0 ){
	    dim3 grid    ((m/BLOCK_SIZE)+1,1,1);
    	dim3 threads (BLOCK_SIZE,1,1);

     	cublasSgemv( handle, cublas_trans_const(MagmaTrans), m, n, &alpha, dC, lddc, dv, 1, &beta, dtau, 1);

 	    stemp<<<  1 ,      1 , 0, queue->cuda_stream()>>>(dC(0,0), dtau, dwork);  
      scnst<<<grid, threads, 0, queue->cuda_stream()>>>(n, dC(0,0), lddc, dtau, dwork);    
  	  sdiff<<<  1 ,      1 , 0, queue->cuda_stream()>>>(dC(0,0), dwork, dwork);  
   	  srow1<<<grid, threads, 0, queue->cuda_stream()>>>(n, dC(0,0), lddc, dtau, dwork);    
    	stmup<<<  n , threads, 0, queue->cuda_stream()>>>(m, dC(0,0), lddc, dtau, dv);    
    	htcns<<<grid, threads, 0, queue->cuda_stream()>>>(m, dv, dtau, dwork);    
   }

    else{
	    dim3 grid    ((m/BLOCK_SIZE),1,1);
    	dim3 threads (BLOCK_SIZE,1,1);
     	
	    cublasSgemv( handle, cublas_trans_const(MagmaTrans), m, n, &alpha, dC, lddc, dv, 1, &beta, dtau, 1);
 
	    stemp<<<  1 ,      1 , 0, queue->cuda_stream()>>>(dC(0,0), dtau, dwork);  
    	scnst<<<grid, threads, 0, queue->cuda_stream()>>>(n, dC(0,0), lddc, dtau, dwork);    
   	  sdiff<<<  1 ,      1 , 0, queue->cuda_stream()>>>(dC(0,0), dwork, dwork);  
    	srow1<<<grid, threads, 0, queue->cuda_stream()>>>(n, dC(0,0), lddc, dtau, dwork);    
    	stmup<<<  n , threads, 0, queue->cuda_stream()>>>(m, dC(0,0), lddc, dtau, dv);    
    	htcns<<<grid, threads, 0, queue->cuda_stream()>>>(m, dv, dtau, dwork);    
   }

    return MAGMA_SUCCESS;
}

//==============================================================================
