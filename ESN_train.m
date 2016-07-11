function [W_out,log] = ESN_train(...
  tr_struct,X,y,W_out)
%% Train the ESN given data
% X is the data and y is the target out
%
% Supported tr_struct.mode values are
% 'normal' (normal eqns given training
%   set. Looks for parameter alph as L2
%   regularlization factor
% 'sgd'    (stochastic gradient descent
%   given start and stop bounds. Looks
%   for parameter alph as L2 reg factor
%   and lambda as learning rate). Sets
%   variable log as the value of the
%   objective over time. Initializes
%   the weights uniformly at random
%   in (-1, 1).
%

%% Setup input parameters
burn_in = tr_struct.burn_in;
Ntrain = tr_struct.Ntrain;
Nepoch = tr_struct.n_epoch;
st_i = burn_in+1;
stop_i = burn_in + Ntrain;
ff = tr_struct.ff;
if ff
  % Concatenate inputs u onto reservoir
  %  state matrix X
  data_N = tr_struct.size_N...
    +tr_struct.size_M;
else
  data_N = tr_struct.size_N;
end
N = tr_struct.size_N;
M = tr_struct.size_M;
L = tr_struct.size_L;

%% Perform training
switch lower(tr_struct.mode)
  case 'normal'
    % Normal equations to calculate
    % optimal weights given training
    % set
    % @TODO: Fix indexing
    W_out = ...
      (y(:,st_i:stop_i) ...
      * X(:,st_i:stop_i)') ...
      / (X(:,st_i:stop_i) ...
      * X(:,st_i:stop_i)' ...
      + tr_struct.alph...
      *eye(size(X,1),size(X,1)));
    log = [];
  case 'sgd'
    % Stochastic gradient descent
    % to train the network online
    lambda = tr_struct.lr;
    alph = tr_struct.alph;
    k = tr_struct.k;
    W_out(1:L,1:data_N) ...
      = (2*rand(L,data_N)-1)*0.01;
    ise = zeros(Ntrain,1);
    
    y_hat = zeros(L,k);
    switch tr_struct.bat_mode
      case 'snapshot'
        step = 1;
      case 'normal'
        step = k;
    end
    keyboard
    for ii = 1:Nepoch
      for i = (st_i+k-1):step:stop_i
        % Different methods for weight
        % updating
        %%% Update the step size
        switch lower(tr_struct.lrn_mode)
          case 'fixed'
            % Uses the local lambda var
            % Uses the same step size
            % on each iteration.
            y_hat = W_out(1:L,1:N) ...
              * X(1:N,(i-k+1):i);
            W_out(1:L,1:N) ...
              = (1-alph*2*lambda) ...
              * W_out(1:L,1:N) ...
              + 2*lambda/k*(y(1:L,(i-k+1):i)...
              - y_hat)*X(1:N,(i-k+1):i).';
          case 'exact'
            % Ignores local lambda var
            % Calculates step size
            % to minimize the RISE
            % along the gradient
            % direction on each cycle
            y_hat = W_out*X(:,i);
            
            grad = 2*(W_out ...
              * X(:,(i-k+1):i)...
              - y(:,(i-k+1):i)) ...
              * X(:,(i-k+1):i).' ...
              + 2*alph*W_out; % accumulate
            
            %           l_star = 1/2 ...
            %             * (2*alph*W_out(:)'*grad(:)...
            %             + 2*(y_hat-y(:,i))'...
            %             * (grad*X(:,i)))...
            %             /((grad*X(:,i))'...
            %             * (grad*X(:,i))...
            %             + alph*grad(:)'*grad(:));
            l_star = norm(grad,'fro').^2 ...
              / norm(grad*X(:,(i-k+1):i),'fro').^2/2;
            
            if l_star < 0
              warning('constraint violated - possibly diverging');
            end
            c = 0;
            decay_factor = exp(-c*(i-st_i));
            l_star = l_star * decay_factor;
            
            W_out =W_out-l_star*grad;
          case 'armillo'
            % Uses two local params
            % Set to powers of two
            % to ease HW implementation
            %
            % Iterate until finding
            % step size that decreases
            % the RISE 'sufficiently',
            % judged by Armillo condition
            % Do this on each cycle
            tau = 0.5;
            c = 0.5;
            slope = -grad*grad.';
            error('Not supported.');
          otherwise
            error('Not supported.');
            
        end
        
        %%% Save the running ISE
        
        ise((i-k-burn_in)/step+1) ...
          = norm(y_hat...
          - y(:,i),2).^2 ...
          + alph*norm(W_out,'fro').^2;
        
      end
    end
    if nargout > 1
      log = ise;
    end
  case 'rls'
    error('Not supported');
    
  otherwise
    error('Not supported.');
end

if 0
  % Diagnostic code dumped here.
  %
  % This logs vector valued W_outs
  % for analysis with Q
  W_log(:,i-st_i+1) = W_out.';
  
  % Q tests each set of intermediate weights' generalization performance
  Q = W_log.'*X(:,stop_i+1:end)  - repmat(y(stop_i+1:end),size(W_log,2),1);
  figure(1);
  plot(diag(Q*Q')); set(gca,'yscale','log');
  figure(2);
  plot(ise); set(gca,'yscale','log');
  keyboard
  disp( min(ise) );
  plot(ise);
  keyboard
  
  % This stores the running output
  % of the ESN during training
  y_log = zeros(Ntrain,1);
  y_log(i-st_i+1) = mean(y_hat);
  
  % This calculates the ISE at each step
  ise_x = norm(y_hat...
    - y(1:L,i:(i+k-1)),2).^2 ...
    + alph*norm(W_out,'fro').^2;
  disp(ise_x); disp(l_star);
  keyboard
end
 