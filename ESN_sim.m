%% Simulate an echo state network
clear all;
close all;
%% Parameters
N = 100;
M = 1;
L = 1;
pr = [0.05];
vvv = [1e3 1e2 1e1 1 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8 1e-9];
alph = 10;
batch_size = 1;
lambda = 1e-4;
sr = [0.1];
distrib = {@rand}; %must be cell                     % name of function for random numbers
nonlin = {@tanh};

%% Test data
% test task - nonlinear transform
T = (pi/3)*1e-2;% make irrational for easy noncyclicality
ppp = 41;       % points per period
Nper = 30;      % num periods
Nt = ppp*Nper;
dt = T*Nper/Nt;
t = 0:dt:T*Nper-dt;
u = (cos(2*pi/T*t));
y = (u.^3)./(max(u.^3));

%% Main loop
% allocate memory
W = zeros(max(N),max(N));
W_out = zeros(max(L),max(N));
W_in = zeros(max(N),max(M));
X = zeros(max(N),Nt);
data = zeros(length(N), length(M), ...
  length(L), length(pr), ...
  length(distrib), length(sr), ...
  length(nonlin), length(alph),...
  length(lambda));

% Programming parameters
w_struct = [];
pr_struct = [];
tr_struct = [];
% USER PARAMS BELOW
tr_struct.burn_in = floor(0.05*Nt);
tr_struct.Ntrain = ...
  floor(Nt/2)- tr_struct.burn_in;
tr_struct.mode = 'sgd';
tr_struct.lrn_mode = 'exact';
tr_struct.bat_mode = 'snapshot';
% END USER PARAMS
tr_i = ...
  tr_struct.burn_in ...
  + tr_struct.Ntrain ...
  + 1;
y_hat = zeros(L, Nt - tr_i + 1);
y_test = y(tr_i:end);
data_sgd = zeros(length(N), length(M), ...
  length(L), length(pr), ...
  length(distrib), length(sr), ...
  length(nonlin), length(alph),...
  length(lambda), length(batch_size), ...
  tr_struct.Ntrain);

for i = 1:length(N)
  for j = 1:length(M)
    for k = 1:length(L)
      w_struct.N = N(i);
      w_struct.M = M(j);
      w_struct.L = L(k);
      tr_struct.size_N = N(i);
      tr_struct.size_L = L(k);
      
      for m = 1:length(pr)
        for n = 1:length(distrib)
          pr_struct.p = pr(m);
          pr_struct.distrib = ...
            distrib{n};
          for o = 1:length(sr)
            % Get ESN
            [W,W_out,W_in]=...
              ESN_init(...
              sr(o),w_struct,...
              pr_struct,W,W_out,...
              W_in);
            % Get states
            for p = 1:length(...
                nonlin)
              X = ESN_evolve(X,...
                W,W_in,u,nonlin{p});
              for q = 1:length(alph)
                % Train output weights
                tr_struct.alph = ...
                  alph(q);
                if strcmp(...
                    tr_struct.mode,...
                    'sgd')
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
                      data_sgd(i,j,k,m,n,o,...
                        p,q,r,s,:) = rise;
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

                else
                  W_out=ESN_train(...
                    tr_struct,X,y,W_out);
                  % Compute predicted
                  % output on test set
                  y_hat(:,:) = ...
                    W_out*X(:,tr_i:end);
                  % Compute scalar data
                  data(i,j,k,m, ...
                    n,o,p,q,1) = ...
                    norm(y_test-y_hat, ...
                    2).^2 / ...
                    length(y_test);
                end
                if ~~(length(alph)-1)
                  fprintf('Done alph, %d of %d\n', ...
                    q, length(alph));
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
  % wipeout W for sparse pattern
%   for q = 1:N
%     for qq = 1:N
%       W(q,qq) = 0;
%     end
%   end
  if ~~(length(N)-1)
    fprintf('Done N, %d of %d\n', ...
      i, length(N));
  end
end

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
  t_test = t(tr_i:end);
  stem(t_test, y_test);
  hold on;
  stem(t_test,y_hat, 'r');
  hold off;
end

% Plot outputs
% t_test = t(tr_i:end);
% plot(t_test, y_test);
% hold on;
% stem(t_test, y_hat, 'r');
% hold off;
% 
% % Compute test MSE
% test_MSE = norm(y_test-y_hat,2).^2 ...
%   /length(y_test);
% fprintf('Test MSE: %1.10f\n', ...
%   test_MSE);