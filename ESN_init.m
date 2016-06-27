function [W, W_out, W_in, W_fb] = ...
  ESN_init(sr,w_struct,pr_struct, ...
            W, W_out, W_in, W_fb)
%% Create weight matrices for ESN
% Pass weights as inputs for in-place
% memory mode
% By: Sam Buchanan

%% Process inputs
if (nargout < 1) || (nargout > 4)
  error('Need to have b/w 1-4 outputs');
end
if (nargin==0)
  error('Need to pass at least SR');
end


if ~exist('w_struct','var')
  % defaults
  N = 1000;
  M = 1;
  L = 1;
  ff = false;
  fb = false;
else
  if ~isfield(w_struct,'N')
    N = 1000;
    % num in reservoir
  else
    N = w_struct.N;
  end
  if ~isfield(w_struct,'M')
    M = 1;
    % num in input
  else
    M = w_struct.M;
  end
  if ~isfield(w_struct,'L')
    L = 1;
    % num in output
  else
    L = w_struct.L;
  end
  if ~isfield(w_struct,'ff')
    ff = false;
    % weight feedforward enabled
  else
    ff = w_struct.ff;
  end
  if ~isfield(w_struct,'fb')
    fb = false;
    % weight feedback enabled
  else
    fb = w_struct.fb;
  end
end

if ~exist('pr_struct','var')
  p = 0.1;
  distrib = 'rand';
else
  if ~isfield(pr_struct,'p')
    p = 0.1;
    % probability a res. weight is
    % nonzero
  else
    p = pr_struct.p;
  end
  if ~isfield(pr_struct,'distrib')
    distrib = @rand;
    % func handle to random distrib to
    % use for weight initializations
  else
    distrib = pr_struct.distrib;
    if ~isequal(class(distrib), ...
        'function_handle')
      error('Pass distrib as func. handle');
    end
  end
end

%% Init main loop
W(1:N,1:N) = ...
  ( rand(N,N)<p )...
  .* (2*distrib(N,N)-1);% reservoir
if trace(W) == 0
  % A nilpotent W has all zero eigenvalues
  % If the spectral radius is zero,
  % the reservoir may not have any computational power
  warning('caution: W may be nilpotent');
end
E = eig(W);
r_sr = max(abs(E));
W = sr .* (W./r_sr);
% NOTE: W_OUT IS INITIALIZED
% IN THE TRAINING FUNCTION

W_in(1:N,1:M) = (2*distrib(N,M)-1);
if fb
  W_fb(1:N,1:L)=(2*distrib(N,L)-1);
end


