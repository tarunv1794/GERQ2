#include "magma_internal.h"
#include "magma_v2.h"

extern "C" magma_int_t
sgeqr2_HT(
    cublasHandle_t handle,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dwork,
    magma_queue_t queue,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*(ldda))

    magma_int_t i, k;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the norms of the trailing columns */
    k = min(m,n);

    for (i = 0; i < k; ++i)
	update_HT(handle, m-i, n-i, dA(i,i), dtau+i, dA(i,i), ldda, dwork, queue);
	
    return *info;
}     
