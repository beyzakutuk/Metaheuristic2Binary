%% ************************************************************************
% About: MadDE Algorithm
% Author: Subhodip Biswas, Debanjan Saha, Shuvodeep De, Adam D Cobb, Swagatam Das and Brian Jalaian
% Cite us: S. Biswas, D. Saha, S. De, A. D. Cobb, S. Das and B. A. Jalaian, 
% "Improving Differential Evolution through Bayesian Hyperparameter Optimization," 
% 2021 IEEE Congress on Evolutionary Computation (CEC), Kraków, Poland, 2021.
%
% For any queries please feel free to contact us
% Subhodip Biswas : sub17was(at)gmail.com
% Debanjan Saha   : debanjansh(at)gmail.com
%% ************************************************************************
% %%%%%%%%%%%     M a d D E      A l g o r i t h m        %%%%%%%%%%%%%%%%%
% Description    This function is used to run the multiple adaptation based
%                Differential Evolution (MadDE) algorithm on the 
%                CEC 2021 Benchmark Suite
% Parameters     ----------------------------------------------------------
% num_runs       : number of runs
% fhd            : cec21 benchmark function handle
% C              : transformations
% problem_size   : dimension
% funcs          : function
% max_nfes       : maximum function evaluations
% optimum        : optima provided as part of CEC 2021 TR
% -------------------------------------------------------------------------
function  [bestSolution, bestFitness, iter] = Binary_MadDE(fhd, dimension, maxIteration, fNumber,T_num)
% load random seeds
mysettings;
problem_size = dimension;
max_nfes = maxIteration;


% algorithm name (underscore) transformation
% initialize boundary limits
lu = [lbArray; ubArray];


%%  MadDE Execution in-progress ...........
    
          
          %% ***  Hyper-parameter settings for MadDEv1.0.0 ***
          %  Below are the set of hyper-parameters used for results
          %  submission at CEC 2021 Single Objective Bound Constraint
          %  Optimization Competition submission at Kraków, Poland, 2021.
          %  --------------------------------------------------------------  
          q_cr_rate = 0.01;                % probability of doing q-best crossover
          p_best_rate = 0.18;              % p_best rate 
          arc_rate = 2.3;                  % archive rate
          memory_size = 10 * problem_size; % memory size 
          pop_size = 2 * problem_size ^ 2; % population size 
          
          max_pop_size = pop_size;         % maximum population size
          min_pop_size = 4.0;              % minimum population size
          
          %% ========= Initialize the main population =====================
          % https://www.mathworks.com/help/stats/sobolset.html
          p = sobolset(problem_size, 'Skip', 1e4, 'Leap', 1e3);
          p = scramble(p, 'MatousekAffineOwen');
          rand0 = net(p, pop_size);
          popold = repmat(lu(1, :), pop_size, 1) + rand0 .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
          pop = popold; % the old population becomes the current population
          nfes = 0;                               %% current evaluation
          bsf_fit_var = 1e+30;                    %% best fitness
          bsf_solution = zeros(1, problem_size);
          
          %%  Calculate functional evaluation on CEC21 Benchmark Suite

          BinaryPop = zeros(size(pop));
          for i = 1:size(pop,1)
              BinaryPop(i,:) = discretizeSolution(pop(i,:), T_num, []);
          end
          fitness = testFunction(BinaryPop', fhd, fNumber);
          %fitness = testFunction(pop', fhd, fNumber);

          fitness = fitness';
          
          %%%%%%%%%% record and store fitness  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
          for i = 1 : pop_size
                nfes = nfes + 1;
                if fitness(i) < bsf_fit_var
                      bsf_fit_var = fitness(i);
                      bsf_solution = pop(i, :);
                end
                if nfes > max_nfes; break; end
          end

          %%%%%%%%%% prob. of each DE operator %%%%%%%%%%%%%%%%%%%%%%%%%%%%

          num_de = 3;                              %% number of operators
          count_S = zeros(1, num_de);
          probDE=1./num_de .* ones(1, num_de);     %% operator probability
          
          memory_sf = 0.2 .* ones(memory_size, 1); %% scaling factor memory
          memory_cr = 0.2 .* ones(memory_size, 1); %% crossover rate memory
          memory_pos = 1;                          
          
          archive.NP = round(arc_rate * pop_size); %% the maximum size of the archive
          archive.pop = zeros(0, problem_size);    %% the solutions stored in te archive
          archive.funvalues = zeros(0, 1);         %% the function value of the archived solutions
          
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          %% ========= main loop ==========================================
          while nfes < max_nfes
                pop = popold; % the old population becomes the current population
                [fitness, sorted_index] = sort(fitness, 'ascend');
                pop=pop(sorted_index,:);
                
                mem_rand_index = ceil(memory_size * rand(pop_size, 1));
                mu_sf = memory_sf(mem_rand_index);
                mu_cr = memory_cr(mem_rand_index);
                
                %% ========= Crossover Rate ===============================
                cr = normrnd(mu_cr, 0.1);
                term_pos = find(mu_cr == -1);
                cr(term_pos) = 0;
                cr = min(cr, 1);
                cr = max(cr, 0);
                
                %% ========= Scaling Factor ===============================
                sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
                pos = find(sf <= 0);
                
                while ~ isempty(pos)
                      sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                      pos = find(sf <= 0);
                end
                
                sf = min(sf, 1);
                
                % initialize mutation population
                r0 = 1 : pop_size;                
                popAll = [pop; archive.pop];
                [r1, r2, r3] = gnR1R2(pop_size, size(popAll, 1), r0);
                vi = zeros(pop_size, problem_size);
                
                %% ========= Mutation Operation ===========================
                bb = rand(pop_size, 1);
                probiter = probDE(1,:);
                l2 = sum(probDE(1:2));
                
                de_1 = bb <= probiter(1)*ones(pop_size, 1);
                de_2 = bb > probiter(1)*ones(pop_size, 1) &  bb <= (l2*ones(pop_size, 1));
                de_3 = bb > l2*ones(pop_size, 1) &  bb <= (ones(pop_size, 1));
                
                pNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions
                randindex = floor(rand(1, pop_size) .* pNP) + 1; %% select from [1, 2, 3, ..., pNP]
                pbest = pop(randindex, :); %% randomly choose one of the top 100p% solutions
                
                % DE/current-to-p-best/1 with archive
                vi(de_1==1, :)  = pop(de_1==1, :) + ...
                      sf(de_1==1, ones(1, problem_size)) .* ...
                      (pbest(de_1==1, :) - pop(de_1==1, :) + pop(r1(de_1==1), :) - popAll(r2(de_1==1), :));
                  
                % DE/current-to-rand/1 with archive
                vi(de_2==1, :) = pop(de_2==1, :) + ...
                      sf(de_2==1, ones(1, problem_size)) .* ...
                      (pop(r1(de_2==1), :) - popAll(r2(de_2==1), :));
                  
                % DE/weighted-rand-to-q-best/1 with attraction
                q_best_rate = 2 * p_best_rate - p_best_rate * (nfes/max_nfes);
                qNP = max(round(q_best_rate * pop_size), 2); %% choose at least two best solutions
                randindex = floor(rand(1, pop_size) .* qNP) + 1; %% select from [1, 2, 3, ..., pNP]
                qbest = pop(randindex, :); %% randomly choose one of the top 100p% solutions
                
                % attracion factor update
                attraction = repmat(0.5 + 0.5 * (nfes/max_nfes), pop_size, problem_size); 
                
                % mutated population
                vi(de_3==1, :) = sf(de_3==1, ones(1, problem_size)) .* ...
                      (pop(r1(de_3==1), :) + ...
                      attraction(de_3==1, :) .* (qbest(de_3==1, :) - pop(r3(de_3==1), :)));
                % bound constrain population within limits
                vi = boundConstraint(vi, pop, lu);
                
                %% ========= q-best binomial crossover ====================
                mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size)); % mask is used to indicate which elements of ui comes from the parent
                rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent
                jrand = sub2ind([pop_size problem_size], rows, cols);
                mask(jrand) = false;
                
                qNP = max(round(q_best_rate * size(popAll, 1)), 2); %% choose at least two best solutions
                randindex = floor(rand(1, size(popAll, 1)) .* qNP) + 1; %% select from [1, 2, 3, ..., qNP]
                popAllbest = popAll(randindex, :); %% randomly choose one of the top 100q% solutions
                popAllbest = popAllbest(1:pop_size, :);
                
                bb = rand(pop_size, 1) <= repmat(q_cr_rate, pop_size, 1);
                qbest = pop;  qbest(bb, :) = popAllbest(bb, :);
                
                ui = vi;      ui(mask) = qbest(mask);
                
                %% ========= Calculate child fitness ======================

                BinaryUi = zeros(size(ui));
                for i = 1:pop_size
                    BinaryUi(i,:) = discretizeSolution(ui(i,:), T_num, BinaryPop(i,:));  % prev olarak BinaryPop(i,:)
                end
        
                children_fitness = testFunction(BinaryUi', fhd, fNumber);
                %children_fitness = testFunction(ui', fhd, fNumber); 

                children_fitness = children_fitness';
                %%%%%%%%%% record and store fitness  %%%%%%%%%%%%%%%%%%%%%%
                  for i = 1 : pop_size
                      nfes = nfes + 1;
                      if children_fitness(i) < bsf_fit_var
                          bsf_fit_var = children_fitness(i);
                          bsf_solution = ui(i, :);                    
                      end
                      if nfes > max_nfes; break; end
                  end

 
                  
                  % Calculate fitness difference
                  dif = abs(fitness - children_fitness);
                  
                  %% ========= Selection ==================================
                  % I == 1: the offspring is better; I == 0: the parent is better
                  I = (fitness > children_fitness);
                  goodCR = cr(I == 1);
                  goodF = sf(I == 1);
                  dif_val = dif(I == 1);
                  
                  archive = updateArchive(archive, popold(I == 1, :), fitness(I == 1));
                  %% ========= update Prob. of each DE ====================
                  diff2 = max(0,(fitness - children_fitness))./abs(fitness);
                  count_S(1)=max(0,mean(diff2(de_1==1)));
                  count_S(2)=max(0,mean(diff2(de_2==1)));
                  count_S(3)=max(0,mean(diff2(de_3==1)));
                  
                  if count_S~=0
                        probDE = max(0.1,min(0.9,count_S./(sum(count_S))));
                  else
                        probDE = 1.0/3 * ones(1,3);
                  end
                  %% ========= update population and fitness ==============
                  [fitness, I] = min([fitness, children_fitness], [], 2);
                  popold = pop;
                  popold(I == 2, :) = ui(I == 2, :);
                  BinaryPop(I == 2, :) = BinaryUi(I == 2, :);

                  num_success_params = numel(goodCR);

                  if num_success_params > 0
                      sum_dif = sum(dif_val);
                      dif_val = dif_val / sum_dif;  

                      %%%% for updating the memory of scaling factor %%%%%%
                      memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);

                      %%%% for updating the memory of crossover rate %%%%%%
                      if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
                        memory_cr(memory_pos)  = -1;
                      else
                        memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
                      end

                      memory_pos = memory_pos + 1;

                      if memory_pos > memory_size
                        memory_pos = 1;
                      end
                  else
                      memory_cr(memory_pos) = 0.5;
                      memory_sf(memory_pos) = 0.5;
                  end

                  %% ========= resizing the population size ===============
                  plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);

                  if pop_size > plan_pop_size
                        reduction_ind_num = pop_size - plan_pop_size;
                        if pop_size - reduction_ind_num <  min_pop_size
                          reduction_ind_num = pop_size - min_pop_size;
                        end

                        pop_size = pop_size - reduction_ind_num;
                        
                        for r = 1 : reduction_ind_num
                          [valBest, indBest] = sort(fitness, 'ascend');
                          worst_ind = indBest(end);
                          popold(worst_ind,:) = [];
                          pop(worst_ind,:) = [];
                          BinaryPop(worst_ind,:) = [];
                          fitness(worst_ind,:) = [];
                        end

                        archive.NP = round(arc_rate * pop_size);

                        if size(archive.pop, 1) > archive.NP
                          rndpos = randperm(size(archive.pop, 1));
                          rndpos = rndpos(1 : archive.NP);
                          archive.pop = archive.pop(rndpos, :);       
                          archive.funvalues = archive.funvalues(rndpos, :);           
                        end
                  end
          end %% maximum evaluations completed
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          % calculate and store error value = fitness - optima
    
              bestSolution = bsf_solution;
              bestFitness = bsf_fit_var;
              iter = nfes;


end %% end of MadDE

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
[dummy IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= archive.NP   % add all new individuals
  archive.pop = popAll;
  archive.funvalues = funvalues;
else                % randomly remove some solutions
  rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
  rndpos = rndpos(1 : archive.NP);
  
  archive.pop = popAll  (rndpos, :);
  archive.funvalues = funvalues(rndpos, :);
end
end


function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)

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
%for i = 1 : inf
for i = 1 : 99999999
    pos = (r1 == r0);
    if sum(pos) == 0
        break;
    else % regenerate r1 if it is equal to r0
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000
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
    if i > 1000
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
    if i > 1000
        error('Can not genrate r3 in 1000 iterations');
    end
end
end


function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang, jingqiao@gmail.com

[NP, D] = size(pop);  % the population size and the problem's dimension

%% check the lower bound
xl = repmat(lu(1, :), NP, 1);
pos = vi < xl;
vi(pos) = (pop(pos) + xl(pos)) / 2;

%% check the upper bound
xu = repmat(lu(2, :), NP, 1);
pos = vi > xu;
vi(pos) = (pop(pos) + xu(pos)) / 2;

end