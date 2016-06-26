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

burn_in = tr_struct.burn_in;
Ntrain = tr_struct.Ntrain;
st_i = burn_in+1;
stop_i = burn_in + Ntrain;
N = tr_struct.size_N;
L = tr_struct.size_L;

switch lower(tr_struct.mode)
  case 'normal'
    % Normal equations to calculate
    % optimal weights given training
    % set
    W_out(1:L,1:N) = ...
      (y(st_i:stop_i) ...
      * X(1:N,st_i:stop_i)') ...
      / (X(1:N,st_i:stop_i) ...
      * X(1:N,st_i:stop_i)' ...
      + tr_struct.alph*eye(N,N));
  case 'sgd'
    lambda = tr_struct.lr;
    alph = tr_struct.alph;
    k = tr_struct.k;
    W_out(1:L,1:N) ...
      = (2*rand(L,N)-1)*0.01;
    ise = zeros(Ntrain,1);
    %     y_log = zeros(Ntrain,1);
    y_hat = zeros(L,k);
    switch tr_struct.bat_mode
      case 'snapshot'
        step = 1;
      case 'normal'
        step = k;
    end
    for i = st_i:step:(stop_i-k+1)
      % Different methods for weight
      % updating
      switch lower(tr_struct.lrn_mode)
        case 'fixed'
          % Uses the local lambda var
          % Uses the same step size
          % on each iteration.
          y_hat = W_out(1:L,1:N) ...
            * X(1:N,i:(i+k-1));
          W_out(1:L,1:N) ...
            = (1-alph*2*lambda) ...
            * W_out(1:L,1:N) ...
            + 2*lambda/k*(y(1:L,i:(i+k-1))...
            - y_hat)*X(1:N,i:(i+k-1)).';
        case 'exact'
          % Ignores local lambda var
          % Calculates step size
          % to minimize the RISE
          % along the gradient
          % direction on each cycle
          y_hat = W_out(1:L,1:N) ...
            * X(1:N,i); % instantaneous
          
          grad = 2/k*(W_out(1:L,1:N) ...
            * X(1:N,i:(i+k-1))...
            - y(1:L,i:(i+k-1))) ...
            * X(1:N,i:(i+k-1)).' ...
            + 2*alph*W_out; % accumulate
          
          l_star = 1/2 ...
            * (2*alph*W_out(:)'*grad(:)...
            + 2*(y_hat-y(1:L,i))'...
            * (grad*X(1:N,i)))...
            /((grad*X(1:N,i))'...
            * (grad*X(1:N,i))...
            + alph*grad(:)'*grad(:));
          if l_star < 0
            warning('constraint violated - possibly diverging');
%             keyboard
          end
          W_out(1:L,1:N) =...
            W_out(1:L,1:N)...
            - l_star * grad;
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
        otherwise
          error('Not supported.');
          
      end
%       W_out(1:L,1:N) ...
%         = (1-alph*2*lambda) ...
%         * W_out(1:L,1:N) ...
%         + 2*lambda/k*(y(1:L,i:(i+k-1))...
%         - y_hat)*X(1:N,i:(i+k-1)).';

      ise_x = norm(y_hat...
        - y(1:L,i:(i+k-1)),2).^2 ...
        + alph*norm(W_out,'fro').^2;
%       disp(ise_x); disp(l_star);
%       keyboard
      
      %       fprintf('sgd: iter %d of %d\n',...
      %         i-st_i+1, Ntrain);
      ise((i-burn_in-1)/step+1) ...
        = norm(y_hat...
        - y(1:L,i),2).^2 ...
        + alph*norm(W_out,'fro').^2;
      %       y_log(i-st_i+1) = mean(y_hat);
      W_log(:,i-st_i+1) = W_out.';
      
    end
    % Q tests each set of intermediate weights' generalization performance
%     Q = W_log.'*X(:,stop_i+1:end)  - repmat(y(stop_i+1:end),size(W_log,2),1);
%     figure(1);
%     plot(diag(Q*Q')); set(gca,'yscale','log');
%     figure(2);
%     plot(ise); set(gca,'yscale','log');
%     keyboard
%     disp( min(ise) );
%     plot(ise);
%     keyboard
    if nargout > 1
      log = ise;
    end
  otherwise
    error('Not supported.');
end
 