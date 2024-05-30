function [getControl] = getControlMPC(N, Q, A, B, C, D, solverstr)
%% Inputs and Outputs
% Inputs: 
    % N - Prediction horizon in optimization 
    % Q - Quantization levels (Constrained set for optimization variables)
    % A, B, C, D - System Matrices
    % solverstr - Solver name.  Can be chosen as follows, 
        % 'gurobi' - Gurobi Optimizer
        % 'mosek' - MOSEK Solver.

 % Outputs: 
    % getControl - returns optimal control (quantization levels) at each
    % prediction horizon ,

%%
nu = 1; % dimension of the control
nx = 3; % dimension of the state
ny = 1; % dimension of the reference signal

% Optimization Variables
u = intvar(repmat(nu,1,N), repmat(1,1,N));      % control 
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));  % state
r = sdpvar(repmat(ny,1,N),repmat(1,1,N));       % reference

%Constraints and Objective Functions
Constraints = [];
Objective = 0;

% Sum over prediction horizon 
for i = 1:N
    et = C*x{i} + D*(u{i}-r{i});   % output/error
    Objective = Objective + et'*et;  % Objective function 
    
    % Constraints
    Constraints = [Constraints, x{i+1} == A*x{i} + B* (u{i}-r{i})]; % State evolution constraints
    Constraints = [Constraints, min(Q) <= u{i} <= max(Q)];          % Input constraints
end

% Parameters input to the optimizer
% Initial state and reference state at each prediction horizon
params_in = {x{1}, [r{:}]};  %
params_out = [u{:}];
% Optimization setting
options = sdpsettings('verbose',0, 'solver', solverstr, 'showprogress',1);
% Optimal Control 
getControl = optimizer(Constraints, Objective, [], params_in, params_out);
end