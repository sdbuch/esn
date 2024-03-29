%% Simulate an echo state network
% @TODO: Refactor the code from this
% inflexible format to a task scheduler
% type of format. A front-end takes
% all the user param settings and
% organizes them into discrete tasks,
% then runs each task through this
% type of loop framework for simulation
clear all;
close all;
%% Parameters
N = [1000];
M = 1;
L = 1;
pr = [0.1];
vvv = [1e3 1e2 1e1 1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9];
alph = 0;
batch_size = [5];
lambda = 1e-2;
sr = [0.8];
distrib = {@rand}; %must be cell
nonlin = {@tanh};

%% Test data
% test task - nonlinear transform
T = 0.02;% make irrational for easy noncyclicality
ppp = 100;       % points per period
Nper = 20;      % num periods
Nt = ppp*Nper;
dt = T*Nper/Nt;
% af = exp(round(log(dt))); % simulate out-of-phase sampling
% dt = dt + af;
t = 0:dt:T*Nper-dt;
% Nt = length(t);
u = (cos(2*pi/T*t)*0.5);
% Example:
% Restructure the input into M channels
Nt = Nt/L(1);
u = reshape(u,M(1),[]);
if size(u,1) ~= M(1)
  error('Reformat input to match input layer parameter M');
end
y = (u.^3);
y = reshape(y(:),L(1),[]);
if size(y,1) ~= L(1)
  error('Reformat output to match output layer parameter L');
end
u = cat(1,u,...
  zeros(max(M)-size(u,1),size(u,2)));
y = cat(1,y,...
  zeros(max(L)-size(y,1),size(y,2)));

%% Setup before main loop
%% Programming parameters
w_struct = [];
pr_struct = [];
tr_struct = [];
%% USER PARAMS BELOW: NETWORK
w_struct.ff = true; %input feedforward to output layer
w_struct.fb = false;%output feedback to reservoir
%% USER PARAMS BELOW: TRAINING
tr_struct.burn_in = floor(0.05*Nt);
tr_struct.Ntrain = ...
  floor(Nt/2)- tr_struct.burn_in;
tr_struct.mode = 'sgd';
tr_struct.lrn_mode = 'exact';
tr_struct.bat_mode = 'normal';
tr_struct.n_epoch = 1e3;
%% END USER PARAMS
%% System params
% allocate memory
if w_struct.ff
  W = zeros(max(N)+max(M),...
    max(N)+max(M));
  W_out = zeros(max(L),max(N)+max(M));
  W_in = zeros(max(N)+max(M),max(M));
  W_fb = zeros(max(N)+max(M),max(L));
else
  W = zeros(max(N),max(N));
  W_out = zeros(max(L),max(N));
  W_in = zeros(max(N),max(M));
  W_fb = zeros(max(N),max(L));
end
data = zeros(length(N), length(M), ...
  length(L), length(pr), ...
  length(distrib), length(sr), ...
  length(nonlin), length(alph),...
  length(lambda));
% internal variables
N = sort(N);
M = sort(M);
L = sort(L);
alph = sort(alph);
lambda = sort(lambda);
batch_size = sort(batch_size);
sr = sort(sr);
tr_i = ...
  tr_struct.burn_in ...
  + tr_struct.Ntrain ...
  + 1;
tr_struct.ff = w_struct.ff;
y_test = y(:,tr_i:end);
data_sgd = zeros(length(N), length(M), ...
  length(L), length(pr), ...
  length(distrib), length(sr), ...
  length(nonlin), length(alph),...
  length(lambda), length(batch_size), ...
  tr_struct.Ntrain);
if tr_struct.ff
  X = zeros(max(M)+max(N),Nt);
else
  X = zeros(max(N),Nt);
end
y_hat = zeros(max(L),Nt-tr_i+1);


%% Main loop
for i = 1:length(N)
  for j = 1:length(M)
    for k = 1:length(L)
      w_struct.N = N(i);
      w_struct.M = M(j);
      w_struct.L = L(k);
      tr_struct.size_N = N(i);
      tr_struct.size_M = M(j);
      tr_struct.size_L = L(k);
  
      for m = 1:length(pr)
        for n = 1:length(distrib)
          pr_struct.p = pr(m);
          pr_struct.distrib = ...
            distrib{n};
          for o = 1:length(sr)
            % Get ESN
            [W,W_out,W_in,W_fb]=...
              ESN_init(...
              sr(o),w_struct,...
              pr_struct,W,W_out,...
              W_in,W_fb);
            % Get states
            for p = 1:length(...
                nonlin)
              % Check to see if we are
              %  running in a feedback
              % or non-feedback mode
              
              if w_struct.fb
                % Feedback execution mode.
                % Process the reservoir
                %  in two sections: training
                %  period (give ideal outputs),
                %  and testing period (give
                %  outputs given trained weights)
                
                % Get training period states
                w_struct.run_idxs...
                  = [1, (tr_struct.Ntrain...
                  + tr_struct.burn_in)];
                
                X = ESN_evolve(X,...
                  W,W_in,u,nonlin{p},...
                  W_fb,y,w_struct);
                for q = 1:length(alph)
                  % Train output weights
                  tr_struct.alph = ...
                    alph(q);
                  for r=1:length(lambda)
                    tr_struct.lr = ...
                      lambda(r);
                    for s = 1:length(batch_size)
                      tr_struct.k = ...
                        batch_size(s);
                      [W_out,rise]=ESN_train(...
                        tr_struct,X,y,W_out);
                      
                      % Compute predicted
                      % output on test set
                      % Because feedback weights
                      % may be used, need to run
                      % in real-time
                      y_hat(:,1) = ...
                        W_out*X(:,tr_i-1);
                      w_struct.run_idxs...
                        = [tr_i, size(X,2)];
                      [X, y_hat]...
                        = ESN_evolve_rt(X,W,W_in,...
                        W_out,u,nonlin{p},W_fb,...
                        y_hat,w_struct);
                      
                      % Compute scalar data
                      data(i,j,k,m, ...
                        n,o,p,q,r,s) = ...
                        norm(y_test-y_hat, ...
                        2).^2 / ...
                        length(y_test);
                      if strcmp(tr_struct.mode,'sgd')
                        data_sgd(i,j,k,m,n,o,...
                          p,q,r,s,:) = rise;
                      end
                      if ~~(length(batch_size)-1)
                        fprintf('Done batch, %d of %d\n', ...
                          s, length(batch_size));
                      end
                    end
                    if ~~(length(lambda)-1)
                      fprintf('Done lambda, %d of %d\n', ...
                        r, length(lambda));
                    end
                  end
                  if ~~(length(alph)-1)
                    fprintf('Done alph, %d of %d\n', ...
                      q, length(alph));
                  end
                end
              else
                
                % non-output-feedback network mode
                % Grab all states at once
                w_struct.run_idxs...
                  = [1, size(X,2)];
                X = ESN_evolve(X,...
                  W,W_in,u,nonlin{p},...
                  W_fb,y,w_struct);
                for q = 1:length(alph)
                  % Train output weights
                  tr_struct.alph = ...
                    alph(q);
                  for r=1:length(lambda)
                    tr_struct.lr = ...
                      lambda(r);
                    for s = 1:length(batch_size)
                      tr_struct.k = ...
                        batch_size(s);
                      [W_out,rise]=ESN_train(...
                        tr_struct,X,y,W_out);
                      % Compute predicted
                      % output on test set
                      
                      y_hat(:,:) = ...
                        W_out*X(:,tr_i:end);
                      % Compute scalar data
                      data(i,j,k,m, ...
                        n,o,p,q,r,s) = ...
                        norm(y_test-y_hat, ...
                        2).^2 / ...
                        length(y_test);
                      if strcmp(tr_struct.mode,'sgd')
                        data_sgd(i,j,k,m,n,o,...
                          p,q,r,s,:) = rise;
                      end
                      if ~~(length(batch_size)-1)
                        fprintf('Done batch, %d of %d\n', ...
                          s, length(batch_size));
                      end
                    end
                    if ~~(length(lambda)-1)
                      fprintf('Done lambda, %d of %d\n', ...
                        r, length(lambda));
                    end
                  end
                  
                  if ~~(length(alph)-1)
                    fprintf('Done alph, %d of %d\n', ...
                      q, length(alph));
                  end
                end
              end
              
              if ~~(length(nonlin)-1)
                fprintf('Done nonlin, %d of %d\n', ...
                  p, length(nonlin));
              end
            end
            if ~~(length(sr)-1)
              fprintf('Done sr, %d of %d\n', ...
                o, length(sr));
            end
          end
          if ~~(length(distrib)-1)
            fprintf('Done distrib, %d of %d\n', ...
              n, length(distrib));
          end
        end
        if ~~(length(pr)-1)
          fprintf('Done pr, %d of %d\n', ...
            m, length(pr));
        end
      end
      if ~~(length(L)-1)
        fprintf('Done L, %d of %d\n', ...
          k, length(L));
      end
    end
    if ~~(length(M)-1)
      fprintf('Done M, %d of %d\n', ...
        j, length(M));
    end
  end
  %  wipeout W for sparse pattern
  for i1 = 1:size(W,1)
    for i2 = 1:size(W,2)
      W(i1,i2) = 0;
    end
  end
  if ~~(length(N)-1)
    fprintf('Done N, %d of %d\n', ...
      i, length(N));
  end
end

%% Process output data hypercube
% In squeeze(data_sgd) with only
% lambda and alpha dims nonsingleton:
% - The rows are diff alpha
% - The cols are diff lambda
sgd_redim = squeeze(data_sgd);
sgd_at_convergence = sgd_redim(:,:,end);
data_redim = squeeze(data);

sgd_at_convergence(sgd_at_convergence ...
  > 100) = 100;
data_redim(data_redim > 100) = 100;


%% Plot outputs
RISE_plots = false;
MSE_plots = false;
MSE_plots_normal = false;
test_plots = true;

if RISE_plots
  % only works if the vector input
  % params are just alph and lambda
  for i = 1:length(alph)
    for j = 1:length(lambda)
      if length(size(sgd_redim)) < 3
        if length(alph) > 1
          rise_ij = squeeze(sgd_redim(i,:));
        elseif length(lambda) > 1
          rise_ij = squeeze(sgd_redim(j,:));
        else
          rise_ij = sgd_redim;
        end
      else
        rise_ij = squeeze(sgd_redim(i,j,:));
      end
      plot( rise_ij );
      set(gca,'yscale','log');
      title(sprintf( ...
        'RISE over iters for alpha=%f,lambda=%f', ...
        alph(i),lambda(j)));
      fprintf('The RISE at end of training was: %f\n', rise_ij(end));
      if length(alph)*length(lambda)>1
        pause
      end
    end
  end
end

if MSE_plots
  figure
  mesh(lambda, alph, data_redim);
  set(gca,'xscale','log');
  xlabel('lambda');
  ylabel('alpha');
  zlabel('test mse');
  set(gca,'yscale','log');
  set(gca,'zscale','log');
end

if MSE_plots_normal
  figure
  plot(alph, data_redim(:,1));
  set(gca,'xscale','log');
  xlabel('alpha');
  ylabel('test mse');
  set(gca,'yscale','log');
end

if test_plots
  figure
  t_test = reshape(t,L,[]);
  t_test = t_test(:,tr_i:end);
  stem(t_test(:), y_test(:));
  hold on;
  stem(t_test(:),y_hat(:), 'r');
  hold off;
end