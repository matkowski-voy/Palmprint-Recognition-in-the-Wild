function B = pls(X,Y,k,gpu)

	Y1 = Y;
	X1 = X;

	W = zeros(size(X,2),k);
	T = zeros(size(X,1),k);
	U = zeros(size(X,1),k);
    if(gpu == true)
        W = gpuArray(W);
        T = gpuArray(T);
        U = gpuArray(U);
        X = gpuArray(X);
        Y = gpuArray(Y);
        X1 = X;
        Y1 = Y;
    end
	for i=1:k
	
		W(:,i) = X'*Y/(Y'*Y);
		W(:,i) = W(:,i)/norm(W);
 		T(:,i) = X*W(:,i);
        
		U(:,i) = Y;
        	P = X'*T(:,i)/(T(:,i)'*T(:,i));
		
		X = X - T(:,i)*P';	
		Y = Y - T(:,i)*T(:,i)'*Y/(T(:,i)'*T(:,i));	
	
	end

    B = X1'*U/(T'*X1*X1'*U)*T'*Y1;

end
