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
      = (2*rand(L,N)-1)*0.001;
    ise = zeros(Ntrain,1);
    %     y_log = zeros(Ntrain,1);
    y_hat = zeros(L,k);
    for i = st_i:k:(stop_i-k+1)
      %       keyboard
      y_hat = W_out(1:L,1:N) ...
        * X(1:N,i:(i+k-1));
      W_out(1:L,1:N) ...
        = (1-alph*2*lambda) ...
        * W_out(1:L,1:N) ...
        + 2*lambda/k*(y(1:L,i:(i+k-1))...
        - y_hat)*X(1:N,i:(i+k-1)).';
      grad = 2/k*(y(1:L,i:(i+k-1))...
        - y_hat)*X(1:N,i:(i+k-1)).';
      slope = -grad*grad.';
      tau = 0.5;
      c = 0.5;
      ise_x = norm(y_hat...
        - y(1:L,i:(i+k-1)),2).^2 ...
        - +alph*norm(W_out,'fro').^2;
      
      keyboard
      
      %       fprintf('sgd: iter %d of %d\n',...
      %         i-st_i+1, Ntrain);
      ise((i-burn_in-1)/k+1) ...
        = norm(y_hat...
        - y(1:L,i:(i+k-1)),2).^2 ...
        + alph*norm(W_out,'fro').^2;
      %       y_log(i-st_i+1) = mean(y_hat);
      W_log(:,i-st_i+1) = W_out.';
      
    end
    % Q tests each set of intermediate weights' generalization performance
    Q = W_log.'*X(:,stop_i+1:end)  - repmat(y(stop_i+1:end),size(W_log,2),1);
    figure(1);
    plot(diag(Q*Q')); set(gca,'yscale','log');
    figure(2);
    plot(ise); set(gca,'yscale','log');
    keyboard
%     disp( min(ise) );
%     plot(ise);
%     keyboard
    if nargout > 1
      log = ise;
    end
  otherwise
    error('Not supported.');
end
 