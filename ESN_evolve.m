function X = ESN_evolve(X,W,W_in,u,...
  nonlin,W_fb,y,w_struct)
%% Evolve states of an ESN given network
%  and data
% Call this twice when using online
%   mode processing - once to get the
%   state data for the training period
%   (supply ideal output as y), and
%   once again during the test period
%   (supply the actual output y_hat 
%    as y)

if ~isequal(class(nonlin), ...
    'function_handle')
  error('Pass nonlin as func. handle');
end

if ~isfield(w_struct,'run_idxs')
  run_idxs = [1 size(X,2)];
else
  run_idxs = w_struct.run_idxs;
end

if length(run_idxs(:))~=2
  error('Pass w_struct.run_idxs as two-value vector');
end

st_i = run_idxs(1);
stop_i = run_idxs(2);
Nt = diff(run_idxs)+1;
N = w_struct.N;
M = w_struct.M;
L = w_struct.L;
if w_struct.ff
  data_N = w_struct.N+w_struct.M;
else
  data_N = w_struct.N;
end

if st_i==1
  % Code for training mode
  X(:,1) = W_in*u(:,1);
  X((N+1):data_N,1) = u(1:M,1);
  if w_struct.fb
    for i = 2:Nt
      X(:,i) = arrayfun(nonlin,...
        W_in*u(:,i) + W*X(:,i-1)...
        + W_fb*y(:,i-1));
      X((N+1):data_N,i) = u(1:M,i);
    end
  else
    for i = 2:Nt
      X(:,i) = arrayfun(nonlin,...
        W_in*u(:,i) + W*X(:,i-1));
      X((N+1):data_N,i) = u(1:M,i);
    end
  end
else
  % Code for test set mode
  if w_struct.fb
    error('Unsupported mode.');
  else
    for i = st_i:stop_i
      X(:,i) = arrayfun(nonlin,...
        W_in*u(:,i) + W*X(:,i-1));
      X((N+1):data_N,i) = u(1:M,i);
    end
    
  end
end