function [bestSolution, bestFitness, iteration]=olshade(fhd, dimension, maxIteration, fNumber)

settings;
Par.n_opr = 4;  %% number of operators 
Par.n     = dimension; %% number of decision vriables
Par.Max_FES = maxIteration;
Par.xmin= lbArray;
Par.xmax= ubArray;
Par.Printing=0;
% Par.f_optimal  = opt(I_fno);
Par.PopSize    = 6*Par.n.^2; 
Par.MinPopSize = 4;

%% Niche Calculation....
Par.nich_size = 10;
[pop_fix,pop_near_idx_fix] = genpop(Par.n,Par.nich_size); % orthogonal array based initialization
Par.pop_fix = pop_fix;
Par.pop_near_idx_fix = pop_near_idx_fix;
Par.Max_FES1 = 0.2*Par.Max_FES;
iter = 0;             %% current generation

%% =================== Define a random seed ===============================
%%-----Becase we ran experiments in parallel, we used "*run" to differentiate
%%-----among runs which started at the same time
% RandStream.setGlobalStream (RandStream('mt19937ar','seed',rand*run));
%% Orthogonal Learning
pop = Par.pop_fix; pop_near_idx = Par.pop_near_idx_fix; 
pop_size = size(pop,1);
current_eval = 0; 
fitness = testFunction(pop', fhd, fNumber)';
%% ====================== store the best ==================
[bestold, bes_l] = min(fitness);     bestx = pop(bes_l,:);
res_det = repmat(bestold,1,pop_size); %% used to record the convergence
%% ==================== to adapt CR and F =================================
hist_pos    = 1;
memory_size = 20*Par.n;
archive_f   = ones(1,memory_size).*0.2;
archive_Cr  = ones(1,memory_size).*0.2;
stop_con    = 0;
while stop_con == 0
    [pop,fitness,pop_near_idx,archive_f,archive_Cr,hist_pos,bestold, bestx, current_eval,res_det] =...
        oDE(pop,fitness,pop_near_idx,archive_f,archive_Cr,hist_pos,bestold, bestx, memory_size...
        , Par.xmin, Par.xmax, current_eval, Par.nich_size,res_det,Par.Printing,fhd, fNumber);
    if current_eval > Par.Max_FES1
        stop_con = 1;
    end
end

[fitness,idx] = sort(fitness);
pop = pop(idx, :);
%% define variable         
PS1 = Par.PopSize;          
%% LSHADE_MODE
EA = pop(1:PS1,:);    EA_obj = fitness(1:PS1);   
%% ===== prob. of each DE operator
probDE1 = 1./Par.n_opr .* ones(1,Par.n_opr);
%% ===================== archive data ====================================
arch_rate   = 2.6;
archive.NP  = arch_rate * PS1; % the maximum size of the archive
archive.pop = zeros(0, Par.n); % the solutions stored in te archive
archive.funvalues = zeros(0, 1); % the function value of the archived solutions
%% ==================== to adapt CR and F =================================
hist_pos=1;
memory_size=20*Par.n;
archive_f= ones(1,memory_size).*0.2;
archive_Cr= ones(1,memory_size).*0.2;
%%
stop_con = 0;  InitPop = PS1;
%% main loop
while stop_con == 0
    iter  = iter+1;
    %% ======================Applying LSHADE_MODE ============================
            %% =============================== Linear Reduction of PS1 ===================================================
            UpdPopSize = round((((Par.MinPopSize - InitPop) / Par.Max_FES) * (current_eval)) + InitPop);
            if PS1 > UpdPopSize
                reduction_ind_num = PS1 - UpdPopSize;
                if PS1 - reduction_ind_num <  Par.MinPopSize
                    reduction_ind_num = PS1 - Par.MinPopSize;
                end
                %% remove the worst ind.
                for r = 1 : reduction_ind_num
                    vv=PS1;
                    EA(vv,:)=[];
                    EA_obj(vv)=[];
                    PS1 = PS1 - 1;
                end
                archive.NP = round(arch_rate * PS1);
                if size(archive.pop, 1) > archive.NP
                    rndpos = randperm(size(archive.pop, 1));
                    rndpos = rndpos(1 : archive.NP);
                    archive.pop = archive.pop(rndpos, :);
                end
            end
%             [EA, EA_obj,probDE1,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,current_eval,res_det] = ...
%                 LSHADE_MODE( EA,EA_obj,probDE1,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,....
%                 Par.xmin, Par.xmax,  Par.n,  PS1,  current_eval, I_fno,res_det,Par.Printing);
            [EA, EA_obj,probDE1,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,current_eval,res_det] = ...
                LSHADE_MODE_greedy( EA,EA_obj,probDE1,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,....
                Par.xmin, Par.xmax,  Par.n,  PS1,  current_eval,res_det,Par.Printing,Par.Max_FES,fhd, fNumber);
   if (current_eval>=Par.Max_FES-4*UpdPopSize)
        stop_con=1;
   end

end
bestSolution=bestx;
bestFitness=bestold;
iteration=1;
end

function [pop,nghbr_idx] = genpop(dim,niche_size)
%dim = 10; Xmin = -100; Xmax = 100;
if dim == 5
   Q = 15;
elseif dim == 10
   Q = 25;
elseif dim == 15
   Q = 40;
elseif dim == 20
   Q = 50;
else 
    Q = dim * 2.5;
end
J = 2;
N = (Q^J - 1)/(Q-1);
Xmax = 100; Xmin = -100;

pop_init = oa_permut(Q,N,J);
pop_init(:,(dim+1):N) = [];

pop = pop_init.*((Xmax-Xmin)/(max(max(pop_init))-min(min(pop_init))))+Xmin;

leng = Q^J;
distanceMatrix = zeros(leng, leng);
nghbr_idx = zeros(leng,niche_size);

% Find neighboring indices

for i = 1:leng
    for j = i+1:leng
        A = pop(i,:); B = pop(j,:);
        distanceMatrix(i,j) = norm(A-B);
        distanceMatrix(j,i)=distanceMatrix(i,j);
    end
    [~,sindex] = sort(distanceMatrix(i,:));
    nghbr_idx(i,:) = sindex(1:niche_size);
end
end

function A = oa_permut(q,n,j)

if n ~= (q^j-1)/(q-1)
    disp('Does not satisfy criteria ..');
    A=[];
    return
end
row=q^j;
col=(q^j-1)/(q-1);
A=zeros(row,col);

% Compute the basic columns
for k = 1:j
    J=((q^(k-1)-1)/(q-1))+1;
    for i = 1:q^j
        A(i,J)=floor(((i-1)/(q^(j-k)))); % I have to use floor to get the correct result
     end  
end

% Compute the non basic columns
for k = 2:j
    J=((q^(k-1)-1)/(q-1))+1;
    for s = 1:J-1
        for t = 1:q-1
            x=J+(s-1)*(q-1)+t;
            A(:,x)=mod(A(:,s)*t+A(:,J),q);
        end
    end
end
A=mod(A,q);
end

function [pop,fitness,pop_near_idx,archive_f,archive_Cr,hist_pos,bestold, bestx, current_eval, res_det] =...
        oDE(pop,fitness,pop_near_idx,archive_f,archive_Cr,hist_pos,bestold, bestx, memory_size,....
        ~, ~, current_eval, nich_size, res_det, Printing,fhd, fNumber)   
    
    %% state
    [pop_size, problem_size] = size(pop);
    mem_rand_index1 = ceil(memory_size * rand(pop_size, 1));
    mu_sf1 = archive_f(mem_rand_index1)';
    mu_cr1 = archive_Cr(mem_rand_index1)';
    
%     %% for generating crossover rate and scale factor
    [cr1,sf1] = gencrsf(mu_cr1,mu_sf1,pop_size);

    children_fitness = zeros(pop_size,1);
    for jj = 1:pop_size
        pop_nghbr = pop(pop_near_idx(jj,:),:);
        fitness_nghbr = [pop_near_idx(jj,:)',fitness(pop_near_idx(jj,:),:)];
        [~,best_mem_idx] = sort(fitness_nghbr(:,2));
        nn = floor(rand*min(nich_size,0.01*pop_size))+1;
        bestmem = pop(fitness_nghbr(best_mem_idx(nn),1),:); % best member
        X1 = DE(pop_nghbr,bestmem,3,sf1(jj),cr1(jj),problem_size,nich_size);
%     %%Evaluate new member
        children_fitness(jj,:) = testFunction(X1', fhd, fNumber);
        current_eval = current_eval+1;
        
        % NSM UYGULANAB?L?R 208
        
       if children_fitness(jj,:) < fitness(jj,:)
           pop(jj,:) = X1;
           fitness(jj,:) = children_fitness(jj,:);
       if fitness(jj,:)<= bestold; bestold = fitness(jj,:); bestx = pop(jj,:); end
       end
    end
        %%% I == 1: the parent is better; I == 2: the offspring is better
        dif = abs(fitness - children_fitness);
        
        % Burdaki k?yaslama NSM skoru ile yap?labilir. 219
        
        I = (fitness > children_fitness);
        goodCR = cr1(I == 1);  
        goodF = sf1(I == 1);
        dif_val = dif(I == 1);
        num_success_params = numel(goodCR);

        if num_success_params > 0 
        sum_dif = sum(dif_val);
        dif_val = dif_val / sum_dif;
      	% for updating the memory of scaling factor 
        archive_f(hist_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);

        % for updating the memory of crossover rate
        if max(goodCR) == 0 || archive_Cr(hist_pos)  == -1
        archive_Cr(hist_pos)  = -1;
        else
        archive_Cr(hist_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
        end

        hist_pos = hist_pos + 1;
        if hist_pos > memory_size;  hist_pos = 1; end
        end
        
%% check to print
if Printing==1
    res_det= [res_det repmat(bestold,1,pop_size)];
end

end  
 
function [cr,sf] = gencrsf(mu_cr,mu_sf,pop_size)
%% for generating crossover rate
cr = normrnd(mu_cr, 0.1);
term_pos = mu_cr == -1;
cr(term_pos) = 0;
cr = min(cr, 1);
cr = max(cr, 0);
      
%% for generating scaling factor
sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
pos = find(sf <= 0);

while ~ isempty(pos)
   sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
   pos = find(sf <= 0);
end
      
sf = min(sf, 1);
end

function ui=DE(pop,bm,st,F,CR,n,NP)
jj = 1;
r1=round(rand*NP); r2=round(rand*NP); r3=round(rand*NP);%r4=round(rand*NP);r5=round(rand*NP);
while (r1==jj || r1==0),r1=ceil(rand*NP);end
while (r2==jj || r2==r1 || r2==0),r2=ceil(rand*NP);end
while (r3==jj || r3==r1 || r3==r2 || r3==0),r3=ceil(rand*NP);end
%while (r4==jj || r4==r1 || r4==r2 || r4==r3 || r4==0),r4=ceil(rand*NP);end
%while (r5==jj || r5==r1 || r5==r2 || r5==r3 || r5==r4 || r5==0),r5=ceil(rand*NP); end
pm1=pop(r1,1:n);
pm2=pop(r2,1:n);
pm3=pop(r3,1:n);
%pm4=pop(r4,1:n);
%pm5=pop(r5,1:n);
popold = pop(jj,:);

mui = rand(1,n) < CR;          % all random numbers < CR are 1, 0 otherwise
if mui==zeros(1,n),nn=randperm(n);mui(nn(1))=1;end
mpo = mui < 0.5;                % inverse mask to mui

if (st == 1)                % DE/rand/1   6
    ui = pm3 + F*(pm1 - pm2);       % differential variation
    ui = popold.*mpo + ui.*mui;     % crossover
% elseif (st == 4)                  % DE/rand/2           10
%     ui = pm5 + F*(pm1 - pm2 + pm3 - pm4);  % differential variation
%     ui = popold.*mpo + ui.*mui;            % crossover
% elseif (st == 5)                  % DE/current-to-best/1    8
%     ui = popold + F*(bm-popold) + F*(pm1 - pm2);        
%     ui = popold.*mpo + ui.*mui;     % crossover
% elseif (st == 6)                  % DE/current-to-best -ve/1    8
%     ui = popold - F*(bm-popold) - F*(pm1 - pm2);        
%     ui = popold.*mpo + ui.*mui;     % crossover
elseif (st == 2)                  % DE/best/1    8
    ui = bm + F*(pm1 - pm2);        
    ui = popold.*mpo + ui.*mui;     % crossover
elseif (st == 3)                   % DE/current-to-best/1    8
    ui = popold + F*(bm-popold) + F*(pm1 - pm2);        
    ui = popold.*mpo + ui.*mui;     % crossover
end

ui=(ui<-100).*(-100)+(ui>=-100).*ui;
ui=(ui>100).*100+(ui<=100).*ui;
end

function [x,fitx,prob,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr,current_eval,res_det] = ...
    LSHADE_MODE_greedy( x,fitx,prob,bestold,bestx,archive,hist_pos,memory_size, archive_f,archive_Cr, xmin, xmax,  n,...
    PopSize,  current_eval,res_det,Printing,Max_FES,fhd, fNumber)

vi=zeros(PopSize,n);

%% calc CR and F
mem_rand_index = ceil(memory_size * rand(PopSize, 1));
mu_sf = archive_f(mem_rand_index);
mu_cr = archive_Cr(mem_rand_index);

%% ========================= generate CR ==================================

cr = normrnd(mu_cr, 0.1);
term_pos = find(mu_cr == -1);
cr(term_pos) = 0;
cr = min(cr, 1);
cr = max(cr, 0);
%% for generating scaling factor


F = mu_sf + 0.1 * tan(pi * (rand(1,PopSize) - 0.5));
pos = find(F <= 0);

while ~ isempty(pos)
    F(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(1,length(pos)) - 0.5));
    pos = find(F <= 0);
end
F = min(F, 1);
F=F';
[fitx,inddd]=sort(fitx);
x=x(inddd,:);
[cr,~]=sort(cr);


%% ======================== generate new x =================================
popAll = [x;archive.pop]; %% set archive
r0 = 1 : PopSize;
%% generate random integer numbers
[r1, r2,r3] = gnR1R2(PopSize, size(popAll, 1), r0);
%% mutation
bb= rand(PopSize, 1);
probiter = prob(1,:);
l2= sum(prob(1:2));
l3= sum(prob(2:3));
op_1 = bb <=  probiter(1)*ones(PopSize, 1);
op_2 = bb > probiter(1)*ones(PopSize, 1) &  bb <= (l2*ones(PopSize, 1)) ;
op_3 = bb > l2*ones(PopSize, 1) &  bb <= (l3*ones(PopSize, 1)) ;
op_4 = bb > l3*ones(PopSize, 1) &  bb <= (1*ones(PopSize, 1)) ;

[~,~]=sort(fitx);

pNP = max(round(0.25* PopSize), 1); %% choose at least two best solutions
randindex = ceil(rand(1, PopSize) .* pNP); %% select from [1, 2, 3, ..., pNP]
randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
phix = x(randindex, :);
vi(op_1==1,:) = x(op_1==1,:)+ F(op_1==1, ones(1, n)) .*( phix(op_1==1,:) - x(op_1==1,:) + x(r1(op_1==1), :) - popAll(r2(op_1==1), :));
vi(op_2==1,:) = x(op_2==1,:)+ F(op_2==1, ones(1, n)) .*( phix(op_2==1,:) - x(op_2==1,:) + x(r1(op_2==1), :) - x(r3(op_2==1), :));
vi(op_4==1,:) = phix(op_4==1,:)+ F(op_4==1, ones(1, n)) .*( x(r3(op_4==1), :) - popAll(r2(op_4==1), :));
%% DE3
pNP = max(round(0.5 * PopSize), 2); %% choose at least two best solutions
randindex = ceil(rand(1, PopSize) .* pNP); %% select from [1, 2, 3, ..., pNP]
randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
phix = x(randindex, :);
vi(op_3==1,:) = F(op_3==1, ones(1, n)).* x(r1(op_3==1), :) + (phix(op_3==1,:) - x(r3(op_3==1), :));

%% handle boundaries
vi = han_boun(vi, xmax, xmin, x,PopSize,2);
%% crossover
if rand<0.4
    for iii=1:PopSize
        mask(iii,:) = rand(1,n) > cr(:, ones(1, n));
    end
    % mask is used to indicate which elements of ui comes from the parent
    rows = (1 : PopSize)'; cols = floor(rand(PopSize, 1) * n)+1; % choose one position where the element of ui doesn't come from the parent
    jrand = sub2ind([PopSize n], rows, cols); mask(jrand) = false;
    ui = vi; 
    ui(mask) = x(mask);
else
    ui=x;
    startLoc= randi(n,PopSize,1);
    for i=1:PopSize
        l=startLoc(i);
        while (rand<cr(i) && l<n)
            l=l+1;
        end
        for j=startLoc(i) : l
            ui(i,j)= vi(i,j);
        end
    end
end
% ui = x; ui(mask) = vi(mask);
%% evaluate
fitx_new = testFunction(ui', fhd, fNumber)';
%% update FITNESS EVALUATIONS
current_eval =current_eval+PopSize;

%% calc. imprv. for Cr and F
diff = abs(fitx - fitx_new);
if current_eval/Max_FES > 0.6
    
% I hesaplan?rken fitness yerine NSM skor kullan?lacak.
% nsmx_new = ui için nsm skoru hesaplanacak.
% I = (nsmx_new < nsmx)
    
I = (fitx_new < fitx);
else
I = SelectionGreedy2(x,ui,fitx(:),fitx_new(:));
end

goodCR = cr(I == 1);
goodF = F(I == 1);

%% ========================= update archive ===============================
archive = updateArchive(archive, x(I == 1, :), fitx(I == 1));
%% ==================== update Prob. of each DE ===========================
diff2 = max(0,(fitx - fitx_new))./abs(fitx);
count_S(1)=max(0,mean(diff2(op_1==1)));
count_S(2)=max(0,mean(diff2(op_2==1)));
count_S(3)=max(0,mean(diff2(op_3==1)));
count_S(4)=max(0,mean(diff2(op_4==1)));
%% update probs.
if count_S~=0
    prob = max(0.1,min(0.9,count_S./(sum(count_S))));
else
    prob = 1/4 * ones(1,4);
end
%% ==================== update x and fitx =================================
fitx(I==1)   = fitx_new(I==1);
x(I == 1, :) = ui(I == 1, :);

%% =================== update memory cr and F =============================

if size(goodF,1)==1
    goodF=goodF';
end
if size(goodCR,1)==1
    goodCR=goodCR';
end
num_success_params = numel(goodCR);
if num_success_params > 0
    weightsDE = diff(I == 1)./ sum(diff(I == 1));
    %% for updating the memory of scaling factor
    archive_f(hist_pos) = (weightsDE' * (goodF .^ 2))./ (weightsDE' * goodF);
    
    %% for updating the memory of crossover rate
    if max(goodCR) == 0 || archive_Cr(hist_pos)  == -1
        archive_Cr(hist_pos)  = -1;
    else
        archive_Cr(hist_pos) = (weightsDE' * (goodCR .^ 2)) / (weightsDE' * goodCR);
    end
    
    hist_pos= hist_pos+1;
    if hist_pos > memory_size;  hist_pos = 1; end
else
    archive_Cr(hist_pos)=0.5;
    archive_f(hist_pos)=0.5;
    % end
end

%% sort new x, fitness
[fitx, ind]=sort(fitx);
x = x(ind,:);
%[min(fitx) max(fitx)]
%% record the best value after checking its feasiblity status
if fitx(1)<bestold  && min(x(ind(1),:))>=-100 && max(x(ind(1),:))<=100
    bestold =fitx(1);
    bestx = x(1,:);
end
%% check to print
if Printing==1
    res_det= [res_det repmat(bestold,1,PopSize)];
end
end

function [r1, r2,r3] = gnR1R2(NP1, NP2, r0)

% gnA1A2 generate two column vectors r1 and r2 of size NP1 & NP2, respectively
%    r1's elements are choosen from {1, 2, ..., NP1} & r1(i) ~= r0(i)
%    r2's elements are choosen from {1, 2, ..., NP2} & r2(i) ~= r1(i) & r2(i) ~= r0(i)
%
% Call: 
%    [r1 r2 ...] = gnA1A2(NP1)   % r0 is set to be (1:NP1)'
%    [r1 r2 ...] = gnA1A2(NP1, r0) % r0 should be of length NP1
%
% Version: 2.1  Date: 2008/07/01
% Written by Jingqiao Zhang (jingqiao@gmail.com)

NP0 = length(r0);

r1 = floor(rand(1, NP0) * NP1) + 1;
% r1 = randperm(NP1,NP0);

%for i = 1 : inf
for i = 1 : 99999999
    pos = (r1 == r0);
    if sum(pos) == 0
        break;
    else % regenerate r1 if it is equal to r0
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = floor(rand(1, NP0) * NP2) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r2 == r1) | (r2 == r0));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r2 in 1000 iterations');
    end
end

r3= floor(rand(1, NP0) * NP1) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r3 == r0) | (r3 == r1) | (r3==r2));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
         r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000, % this has never happened so far
        error('Can not genrate r2 in 1000 iterations');
    end
end
end

function x = han_boun (x, xmax, xmin, x2, PopSize,~)
hb=randi(3);
switch hb
    case 1 % for DE
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x(pos) = (x2(pos) + x_L(pos)) / 2;
        
        x_U = repmat(xmax, PopSize, 1);
        pos = x > x_U;
        x(pos) = (x2(pos) + x_U(pos)) / 2;
        
    case 2 
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x_U = repmat(xmax, PopSize, 1);
        x(pos) = min(x_U(pos),max(x_L(pos),2*x_L(pos)-x2(pos)))  ;
        pos = x > x_U;
        x(pos) = max(x_L(pos),min(x_U(pos),2*x_L(pos)-x2(pos)));
        
   case 3 
        x_L = repmat(xmin, PopSize, 1);
        pos = x < x_L;
        x_U = repmat(xmax, PopSize, 1);
        x(pos) = x_L(pos)+ rand*(x_U(pos)-x_L(pos) ) ;
        pos = x > x_U;
        x(pos) = x_L(pos)+ rand*(x_U(pos)-x_L(pos));
        
end  
end

function archive = updateArchive(archive, pop, funvalue)
% Update the archive with input solutions
%   Step 1: Add new solution to the archive
%   Step 2: Remove duplicate elements 
%   Step 3: If necessary, randomly remove some solutions to maintain the archive size
%
% Version: 1.1   Date: 2008/04/02
% Written by Jingqiao Zhang (jingqiao@gmail.com)

if archive.NP == 0, return; end

if size(pop, 1) ~= size(funvalue,1), error('check it'); end

% Method 2: Remove duplicate elements
popAll = [archive.pop; pop ];
funvalues = [archive.funvalues; funvalue ];
[~, IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= archive.NP   % add all new individuals
  archive.pop = popAll;
  archive.funvalues = funvalues;
else                % randomly remove some solutions
  rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
  rndpos = rndpos(1 : ceil(archive.NP));
  
  archive.pop = popAll  (rndpos, :);
  archive.funvalues = funvalues(rndpos, :);
end
end

function I = SelectionGreedy2(x,ui,fitx,fitx_new)
N = size(x,1);
n = max(ceil(N*0.15),1);
fit = [fitx;fitx_new];
if rand < 1
I1 = false(N,1);
for i = 1:N
    ind = randperm(2*N,n);
    neigh = fit(ind(1:n));
    imp   = sum(neigh > fitx_new(i))/n;
    if imp > 0.25
        I1(i) = true;
    end
end
else
    I1 = true(N,1);
end
I2 = (fitx_new < fitx);
I = I1 & I2;
end

