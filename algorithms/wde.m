function [bestSolution, bestFitness, iteration]=wde(fhd, dimension, maxIteration, fNumber)

settings;

N = 50; 
D = dimension;
low = lbArray;
up = ubArray;
MaxEpk = ceil(maxIteration/N);

%INITIALIZATION
if numel(low)==1, low = low * ones(1,D); up = up * ones(1,D); end % this line must be adapted to your problem
P = GenP(2*N,D,low,up); % see Eq.1 in [1]

fitP = testFunction(P', fhd, fNumber);
% ------------------------------------------------------------------------------------------
for epk=1:MaxEpk
    
    j = randperm(2*N);
    k = j(1:N);
    l = j(N+1 : 2*N);
    trialP = P(k,:);
    fitTrialP = fitP(k);    
    
    temp = trialP ;  % memory
    for index = 1:N
        w = rand(N,1).^3;
        w = w ./ sum(w);
        tempP = P(l,:);
        res = zeros(N, D);
        for i = 1 : N 
            res(i,:) = w(i) * tempP(i,:);
        end
        temp(index,:) = sum(res);
    end
    
    while 1, m = randperm(N); if sum(1:N == m, 2)==0 ==0, break; end, end
    
    E = temp - trialP(m,:) ;
    
    %  recombination
    M = GenM(N,D);
    
    if rand<rand, F = randn(1,D).^3 ; else F = randn(N,1).^3 ; end
    
    Trial = zeros(N, D);
    if numel(F) == N
        for i = 1 : N
            Trial(i, :) = trialP(i, :) + F(i) .* M(i, :) .* E(i, :);
        end    
    else
        for i = 1 : D
            Trial(:, i) = trialP(:, i) + F(i) .* M(:, i) .* E(:, i);
        end 
    end
    Trial = BoundaryControl(Trial,low,up) ; % see Algorithm-3 in [1]
    
    fitT = testFunction(Trial', fhd, fNumber);
    ind = fitT < fitTrialP ;
    
    trialP(ind,:)  = Trial(ind,:) ;
    fitTrialP(ind) = fitT(ind) ;
    
    fitP(k)=fitTrialP;
    P(k,:)=trialP;
    
    % keep the solutions
   
    [bestsol,ind] = min(fitP);
    best = P(ind,:);
  
end %epk

    bestSolution=best;
    bestFitness=bestsol;
    iteration=epk*N;
return
function M = GenM(N,D)
M = zeros(N,D);
for i=1:N
    if rand<rand, k = rand^3;  else k=1-rand^3; end
    V = randperm(D);
    j = V( 1:ceil(k*D) );
    M(i,j) =  1;
end
function pop = GenP(N,D,low,up)
pop = ones(N,D);
for i = 1:N
    for j = 1:D
        pop(i,j) = rand * ( up(j) - low(j) ) + low(j);
    end
end
return
function pop = BoundaryControl(pop,low,up)
[popsize,dim] = size(pop);
for i = 1:popsize
    for j = 1:dim
        F = rand.^3 ;
        if pop(i,j) < low(j), pop(i,j) = low(j) +  F .* ( up(j)-low(j) );  end
        if pop(i,j) > up(j),  pop(i,j) = up(j)  +  F .* ( low(j)-up(j));   end
    end
end
return