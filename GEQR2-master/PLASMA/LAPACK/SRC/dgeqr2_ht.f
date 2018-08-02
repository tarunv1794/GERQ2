	Subroutine DGEQR2_HT(M, N, A, LDA, TAU, WORK, INFO)
	
!------ Input Variables	-------!
	Integer M, N, LDA, INFO 
	Double Precision A(LDA,*), WORK(*), TAU(*) 
	
!------ Local Variables -------!
	Integer I, L

!------ Internal Functions Call ------!
	Intrinsic MAX

!------ External Subroutine Call ------!
	External XERBLA, UPDATE1
	
!------ Intializing Variables ------!
	Tau(1:N) = 0

!------ Checking for Parameters ------!

	INFO = 0
	If( M.LT.0 ) Then
	INFO = -1
	Else If( N.LT.0 ) Then
     	INFO = -2
      	Else If( LDA.LT.MAX( 1, M ) ) Then
        INFO = -4
     	End If
      	If( INFO.NE.0 ) Then
        Call XERBLA( 'DGEQR2_HT', -INFO )
        Return
      	End If

	
	Do I = 1,N  	

	L = M-I+1

	Call UPDATE1(L, A(I:M,I), A, LDA, I, M, N, TAU(I)) 	

	End Do

	Return
	End Subroutine
