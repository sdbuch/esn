function X = ESN_evolve(X,W,W_in,u,...
  nonlin)
%% Evolve states of an ESN given network
%  and data

if ~isequal(class(nonlin), ...
    'function_handle')
  error('Pass nonlin as func. handle');
end

N = size(W,1);
Nt = size(X,2);
X(:,1) = W_in*u(1);
for i = 2:Nt
  X(:,i) = arrayfun(nonlin,...
    W_in*u(i) + W*X(:,i-1));
end
