function [bestSolution, bestFitness, iteration]=bsa(fhd, dimension, maxIteration, fNumber)

settings;

popsize = 30; DIM_RATE = 1; dim = dimension;
%INITIALIZATION
low=lbArray ;
up=ubArray;
pop=GeneratePopulation(popsize,dim,low,up); % see Eq.1 in [1]

fitnesspop = testFunction(pop', fhd, fNumber);

historical_pop=GeneratePopulation(popsize,dim,low,up); % see Eq.2 in [1]

% historical_pop  is swarm-memory of BSA as mentioned in [1].

% ------------------------------------------------------------------------------------------
fe=0;
    while fe<maxIteration

        %SELECTION-I
        if rand<rand, historical_pop=pop; end  % see Eq.3 in [1]
        historical_pop=historical_pop(randperm(popsize),:); % see Eq.4 in [1]
        F=get_scale_factor; % see Eq.5 in [1], you can other F generation strategies 
        map=zeros(popsize,dim); % see Algorithm-2 in [1]         
        if rand<rand
            for i=1:popsize,  u=randperm(dim); map(i,u(1:ceil(DIM_RATE*rand*dim)))=1; end
        else
            for i=1:popsize,  map(i,randi(dim))=1; end
        end
        % RECOMBINATION (MUTATION+CROSSOVER)   
        offsprings=pop+(map.*F).*(historical_pop-pop);   % see Eq.5 in [1]    
        offsprings=BoundaryControl(offsprings,low,up); % see Algorithm-3 in [1]
        % SELECTON-II
        
        fitnessoffsprings = testFunction(offsprings', fhd, fNumber);      

        fe=fe+popsize;
        ind=fitnessoffsprings<fitnesspop;
        fitnesspop(ind)=fitnessoffsprings(ind);
        pop(ind,:)=offsprings(ind,:);
        [globalminimum,ind]=min(fitnesspop);    
        globalminimizer=pop(ind,:);
        % EXPORT SOLUTIONS
        bestSolution = globalminimizer;
        bestFitness = globalminimum;
    end
    iteration = fe;
return

function pop=GeneratePopulation(popsize,dim,low,up)
pop=ones(popsize,dim);
for i=1:popsize
    for j=1:dim
        pop(i,j)=rand*(up(j)-low(j))+low(j);
    end
end
return

function pop=BoundaryControl(pop,low,up)
[popsize,dim]=size(pop);
for i=1:popsize
    for j=1:dim                
        k=rand<rand; % you can change boundary-control strategy
        if pop(i,j)<low(j), if k, pop(i,j)=low(j); else pop(i,j)=rand*(up(j)-low(j))+low(j); end, end        
        if pop(i,j)>up(j),  if k, pop(i,j)=up(j);  else pop(i,j)=rand*(up(j)-low(j))+low(j); end, end        
    end
end
return

function F=get_scale_factor % you can change generation strategy of scale-factor,F    
     F=3*randn; % STANDARD brownian-walk
    % F=4*randg;  % brownian-walk    
    % F=lognrnd(rand,5*rand);  % brownian-walk              
    % F=1/normrnd(0,5);        % pseudo-stable walk (levy-like)
    % F=1./gamrnd(1,0.5);      % pseudo-stable walk (levy-like, simulates inverse gamma distribution; levy-distiribution)   
return