%% Simulate an echo state network
% fixed point parameters
w = 16;     % word length
s = 1;       % signed numbers
frac_bits_learn_rate = 6;
q_w = quantizer('fixed', 'floor', 'Saturate', [w w-1]);

% params
N = 1000;
M = 1;
L = 1;
p = 0.1;
alph = ufi(1e-3,w,frac_bits_learn_rate); % allow up to 1e3... tuneable
sr = 0.8;
nonlin = 'tanh';



% Network
W = fi(randquant(q_w,10), s, w, w-1);
sp_pat = ( rand(10,10)<0.1 );
for i = 1:10
    for j = 1:10
        if ~sp_pat(i,j)
            W(i,j) = fi(0,s,w,w-1); 
        end
    end
end
E = eig((double(W)));
r_sr = max(abs(E));
W = sr .* (W./r_sr);
W_in = (2* eval( strcat(distrib,'(N,M)') ) - 1);
W_out = 2*rand(L,N)-1;                % initialize output weights

% test data - nonlinear transform
T = pi*1e-2;
ppp = 40;
Nper = 10;
Nt = ppp*Nper;
dt = T*Nper/Nt;
t = 0:dt:T*Nper-dt;
u = (cos(2*pi/T*t)*0.5).';
y = (u.^3).';

% evolve some states
X = zeros(N,Nt);
X(:,1) = W_in*u(1);
for i = 2:Nt
  X(:,i) = eval(strcat(nonlin, '( W_in*u(i-1) + W*X(:,i-1) )'));
end

% Train a linear classifier on some of the states
burn_in = floor(0.05*length(t));      % discard first <burn_in> cycles
Ntrain = floor(length(t)/2)-burn_in;  % number of samples to train
st_i = burn_in+1;
stop_i = burn_in+Ntrain+1;
X_train = X(:,st_i:stop_i);
y_train = y(st_i:stop_i);

P = y_train*X_train.';
R = X_train*X_train.'+eye(N)*alph;    % L2 regularized
W_out = P*R^-1;

% Test the readout on the rest of the data
X_test = X(:,stop_i+1:end);
y_test = y(stop_i+1:end);
y_hat = zeros(size(y_test));
for i = 1:size(X_test,2)
  y_hat(i) = W_out*X_test(:,i);
end

% Plot outputs
t_test = t(stop_i+1:end);
plot(t_test, y_test);
hold on;
stem(t_test, y_hat, 'r');
hold off;

% Compute test MSE
test_MSE = norm(y_test-y_hat,2).^2/length(y_test);
fprintf('Test MSE: %1.10f\n', test_MSE);