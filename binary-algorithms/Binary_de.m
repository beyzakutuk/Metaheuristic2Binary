function [bestSolution, bestFitness, iteration]=Binary_de(fhd, dimension, maxIteration, fNumber, T_num)

mysettings;

nVar=dimension;            % Number of Decision Variables
VarSize=[1 nVar];   % Decision Variables Matrix Size
VarMin=lbArray;          % Lower Bound of Decision Variables
VarMax=ubArray;          % Upper Bound of Decision Variables

% DE Parameters
nPop=50;        % Population Size
MaxIt=ceil(maxIteration/nPop);      % Maximum Number of Iterations

beta_min=0.2;   % Lower Bound of Scaling Factor
beta_max=0.8;   % Upper Bound of Scaling Factor
pCR=0.2;        % Crossover Probability

% Initialization

empty_individual.Position=[];
empty_individual.Cost=[];

BestSol.Cost=inf;
pop=repmat(empty_individual,nPop,1);
BinaryPop = zeros(nPop, nVar);

for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);

    BinaryPop(i,:) = discretizeSolution(pop(i).Position, T_num, []);
    pop(i).Cost = testFunction(BinaryPop(i,:)', fhd, fNumber); % BURAYI GÜNCELLEDİK!!!
    
    %disp(['Binary çözüm: ', mat2str(BinaryPop(i,:))]);
    %pop(i).Cost=testFunction(pop(i).Position', fhd, fNumber);
    if pop(i).Cost<BestSol.Cost
        BestSol=pop(i);
    end 
end

% DE Main Loop
for it=1:MaxIt 
    for i=1:nPop
        x=pop(i).Position;
        A=randperm(nPop);
        A(A==i)=[];
        a=A(1);
        b=A(2);
        c=A(3);
        
        % Mutation
        %beta=unifrnd(beta_min,beta_max);
        beta=unifrnd(beta_min,beta_max,VarSize);
        y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);
        y = max(y, VarMin);
		y = min(y, VarMax);
		
        % Crossover
        z=zeros(size(x));
        j0=randi([1 numel(x)]);
        for j=1:numel(x)
            if j==j0 || rand<=pCR
                z(j)=y(j);
            else
                z(j)=x(j);
            end
        end
        
        NewSol.Position=z;

        binarySol = discretizeSolution(NewSol.Position, T_num, BinaryPop(i,:));
        %disp(['Binary çözüm: ', mat2str(binarySol)]);
        NewSol.Cost = testFunction(binarySol', fhd, fNumber); % BURAYI GÜNCELLEDİK!!!

        %NewSol.Cost=testFunction(NewSol.Position', fhd, fNumber);
        
        if NewSol.Cost<pop(i).Cost
            pop(i)=NewSol;
            BinaryPop(i,:) = binarySol;
            
            if pop(i).Cost<BestSol.Cost
               BestSol=pop(i);
            end
        end
        
    end
end
bestSolution=BestSol.Position;
bestFitness=BestSol.Cost;
iteration=it*nPop;
end

