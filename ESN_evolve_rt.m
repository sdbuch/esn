function [X, y] = ESN_evolve_rt(X,W,...
  W_in,W_out,u,nonlin,W_fb,y,w_struct)
%% Real time State Evolution for ESN
% Call this function instead of ESN_evolve
% when using a trained model with
% feedback weights

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

% Code for test set mode
if w_struct.fb
  for i = st_i:stop_i
    output_i = i-st_i+2;
    X(:,i) = arrayfun(nonlin,...
      W_in*u(:,i) + W*X(:,i-1)...
      + W_fb*y(:,output_i-1));
    X((N+1):data_N,i) = u(1:M,i);
    y(:,output_i-1) = ...
      W_out*X(:,i);
  end
else
  for i = st_i:stop_i
    output_i = i-st_i+2;
    X(:,i) = arrayfun(nonlin,...
      W_in*u(:,i) + W*X(:,i-1)...
      + W_fb*y(:,output_i-1));
    X((N+1):data_N,i) = u(1:M,i);
    y(:,output_i-1) = ...
      W_out*X(:,i);
  end
end