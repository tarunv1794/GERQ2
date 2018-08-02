#include "lapacke_utils.h"

lapack_int LAPACKE_dgeqr2_ht( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau)
{
    lapack_int info = 0;
    double* work = NULL;
    if( matrix_order != LAPACK_COL_MAJOR && matrix_order != LAPACK_ROW_MAJOR ) {
        LAPACKE_xerbla( "LAPACKE_dgeqr2_ht", -1 );
        return -1;
    }
#ifndef LAPACK_DISABLE_NAN_CHECK
    /* Optionally check input matrices for NaNs */
    if( LAPACKE_dge_nancheck( matrix_order, m, n, a, lda ) ) {
        return -4;
    }
#endif
    /* Allocate memory for working array(s) */
    work = (double*)LAPACKE_malloc( sizeof(double) * MAX(1,n) );
    if( work == NULL ) {
        info = LAPACK_WORK_MEMORY_ERROR;
        goto exit_level_0;
    }
    /* Call middle-level interface */
    info = LAPACKE_dgeqr2_ht_work( matrix_order, m, n, a, lda, tau, work );
    /* Release memory and exit */
    LAPACKE_free( work );
exit_level_0:
    if( info == LAPACK_WORK_MEMORY_ERROR ) {
        LAPACKE_xerbla( "LAPACKE_dgeqr2_ht", info );
    }
    return info;
}
