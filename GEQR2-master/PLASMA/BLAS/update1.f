	Subroutine UPDATE1(L, X, A, LDA, K, M, N, TAU)

!------ Input Variables	-------!
	Integer L, K, N, M, LDA 
	Double Precision X(L), A(LDA, * ), Tau
	
!------ Local Variables -------!
	Integer I, J
	Double Precision Norm, S, B, Beta
	Double Precision DNRM2,DDOT

!------ Internal Functions Call ------!
	Intrinsic SQRT, SIGN
	External DNRM2,DDOT
	
!------ Intializing Variables 	
	Norm = 0
	S = 0
	B = 0

!------ Calculating Norm ------!

	Norm = -SIGN( DNRM2(L,X,1), X(1) )
	Beta = ( X(1) - Norm )
	Tau  = -Beta/Norm 

!----- Calculating Updated-Matrix for one column ------!
	
	If(K .NE. N) Then

	Do I = K+1,N

	B = A(K,I)*Beta	
 	S = DDOT(L-1,X(2:L),1,A(K+1:M,I),1)
	B = B+S	
	B = B/(Norm*Beta)

	A(K,I) = A(K,I) + (Beta*B)	

	Do J = K+1,M 

 	A(J,I) = A(J,I) + (A(J,K)*B)
	
	End Do
     

	End Do
		
	End If

	X(1) = Norm

	If (L .GE. 2) Then	
		
	Do I = 2,L
	
	X(I) = X(I)/Beta	
		
	End Do

	End If

		
	Return
	End Subroutine UPDATE1
