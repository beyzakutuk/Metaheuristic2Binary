function [bestSolution, bestFitness, iteration]=Binary_bes(fhd, dimension, maxIteration, fNumber,T_num)

mysettings;
nPop=100;
MaxIt=(maxIteration/(nPop*3))+1;
low = lbArray;
high = ubArray;
dim = dimension;
BestSol.cost = inf;
for i=1:nPop
     pop.pos(i,:) = low+(high-low).*rand(1,dim);

     pop.BinaryPos(i,:) = discretizeSolution(pop.pos(i,:), T_num, []);
     pop.cost(i)=testFunction(pop.BinaryPos(i,:)', fhd, fNumber);

      %BinaryPop = zeros(size(pop));
        %for i = 1:size(pop,1)
        %    BinaryPop(i,:) = discretizeSolution(pop(i,:), T_num, []);
       % end
    
      %  fitness = testFunction(BinaryPop', fhd, func);



     %pop.cost(i)=testFunction(pop.pos(i,:)', fhd, fNumber);
    if pop.cost(i) < BestSol.cost
        BestSol.pos = pop.pos(i,:);
        BestSol.cost = pop.cost(i);
    end
end
for t=1:MaxIt
    %%               1- select_space 
    [pop, BestSol, ~]=select_space(pop,nPop,BestSol,low,high,dim,fhd,fNumber,T_num);
    %%                2- search in space
    [pop, BestSol, ~]=search_space(pop,BestSol,nPop,low,high,fhd,fNumber,T_num);
    %%                3- swoop
  [pop, BestSol, ~]=swoop(pop,BestSol,nPop,low,high,fhd,fNumber,T_num);
end
bestSolution=BestSol.pos;
bestFitness=BestSol.cost;
iteration=1;
end

function [pop, BestSol, s1]=select_space(pop,npop,BestSol,low,high,dim,fhd,fNumber,T_num)
Mean=mean(pop.pos);
% Empty Structure for Individuals
empty_individual.pos = [];
empty_individual.BinaryPop = [];
empty_individual.cost = [];
lm= 2;
s1=0;
for i=1:npop
    newsol=empty_individual;
    newsol.pos= BestSol.pos+ lm*rand(1,dim).*(Mean - pop.pos(i,:));
    newsol.pos = max(newsol.pos, low);
    newsol.pos = min(newsol.pos, high);

    newsol.BinaryPop = discretizeSolution(newsol.pos, T_num, []);

    % Değerlendir
    newsol.cost = testFunction(newsol.BinaryPop', fhd, fNumber);
    %newsol.cost=testFunction(newsol.pos', fhd, fNumber);

    if newsol.cost<pop.cost(i)
       pop.pos(i,:) = newsol.pos;
       pop.BinaryPop(i,:) = newsol.BinaryPop;
       pop.cost(i)= newsol.cost;
       s1=s1+1;
         if pop.cost(i) < BestSol.cost
          BestSol.pos= pop.pos(i,:);
         BestSol.cost=pop.cost(i); 
         end
    end
end
end

function [pop, best, s1]=search_space(pop,best,npop,low,high,fhd,fNumber,T_num)
Mean=mean(pop.pos);
a=10;
R=1.5;
% Empty Structure for Individuals
empty_individual.pos = [];
empty_individual.BinaryPop = [];
empty_individual.cost = [];
s1=0;
for i=1:npop-1
    A=randperm(npop);
pop.pos=pop.pos(A,:);
pop.cost=pop.cost(A);
        [x, y]=polr(a,R,npop);
    newsol=empty_individual;
   Step = pop.pos(i,:) - pop.pos(i+1,:);
   Step1=pop.pos(i,:)-Mean;
   newsol.pos = pop.pos(i,:) +y(i)*Step+x(i)*Step1;
    newsol.pos = max(newsol.pos, low);
    newsol.pos = min(newsol.pos, high);

    newsol.BinaryPop = discretizeSolution(newsol.pos, T_num, []);

    % Fitness değerlendirme
    newsol.cost = testFunction(newsol.BinaryPop', fhd, fNumber);
    %newsol.cost=testFunction(newsol.pos', fhd, fNumber);

    if newsol.cost<pop.cost(i)
       pop.pos(i,:) = newsol.pos;
       pop.BinaryPop(i,:) = newsol.BinaryPop;
       pop.cost(i)= newsol.cost;
        s1=s1+1;
        if pop.cost(i) < best.cost
            best.pos= pop.pos(i,:);
            best.cost=pop.cost(i); 
        end
    end
end
end

function [pop, best, s1]=swoop(pop,best,npop,low,high,fhd,fNumber,T_num)
Mean=mean(pop.pos);
a=10;
R=1.5;
% Empty Structure for Individuals
empty_individual.pos = [];
empty_individual.BinaryPop = [];
empty_individual.cost = [];
s1=0;
for i=1:npop
    A=randperm(npop);
pop.pos=pop.pos(A,:);
pop.cost=pop.cost(A);
        [x, y]=swoo_p(a,R,npop);
    newsol=empty_individual;
   Step = pop.pos(i,:) - 2*Mean;
   Step1= pop.pos(i,:)-2*best.pos;
   newsol.pos = rand(1,length(Mean)).*best.pos+x(i)*Step+y(i)*Step1;
    newsol.pos = max(newsol.pos, low);
    newsol.pos = min(newsol.pos, high);

    newsol.BinaryPop = discretizeSolution(newsol.pos, T_num, []);

    % Binary çözüm üzerinden fitness hesapla
    newsol.cost = testFunction(newsol.BinaryPop', fhd, fNumber);
   %newsol.cost=testFunction(newsol.pos', fhd, fNumber);

    if newsol.cost<pop.cost(i)
       pop.pos(i,:) = newsol.pos;
        pop.BinaryPop(i,:) = newsol.BinaryPop;
       pop.cost(i)= newsol.cost;
        s1=s1+1;
        if pop.cost(i) < best.cost
            best.pos= pop.pos(i,:);
            best.cost=pop.cost(i); 
        end
    end
end
end

function [xR, yR]=swoo_p(a,~,N)
th = a*pi*exp(rand(N,1));
r  =th; %R*rand(N,1);
xR = r.*sinh(th);
yR = r.*cosh(th);
xR=xR/max(abs(xR));
yR=yR/max(abs(yR));
end
 
 function [xR, yR]=polr(a,R,N)
%// Set parameters
th = a*pi*rand(N,1);
r  =th+R*rand(N,1);
xR = r.*sin(th);
yR = r.*cos(th);
xR=xR/max(abs(xR));
yR=yR/max(abs(yR));
end