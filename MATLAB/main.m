% MPC Setup in YALMIP
yalmip('clear')
close all
clear
clc

%% Qunatization Parameters 
nob = 16;        % number of bits
lvl = 2^nob;    % total number of quantization levels
Q = 0:1:lvl-1;  % quantization levels

%% Filter Parameters
Fs = 1e6;       % Sampling frequency
Ts = 1/Fs;
Fc = 1e5;       % Cutoff frequency
Wn = Fc/(Fs/2); 
switch 1
    case 1
        % Butterworth Filter
        [b, a] = butter(3, Wn);     % Filter parameters
        [h,tb] = impz(b,a, 100);    % Impulse response of the filter
    case 2
        b =  [1, 0.91, 0];
        a = [1, -1.335, 0.664];
    case 3
        z = tf('z',Ts);
        [b1, a1] = butter(1, Wn);     % Filter parameters
        W = tf(b1,a1, Ts);
        F = (W)*1/z;
        H = minreal(1/(1-F), 1e-5);
        b = [H.num{1}];
        a = [H.den{1}];

end
[A,B,C,D]= tf2ss(b,a);      % Transfer function to state space

%% Reference Signal
Ts = 1/Fs;          % Sampling time
t_end = 0.01;          % Simulation time
t = 0:Ts:t_end;     
fs = 3;             % Signal frequency
As = lvl;           % Singal Amplitude
t_new = 0:Ts:t_end+0.0001;
ref = ((1/2)*((As-1)*sin(2*pi*fs*t_new)) + (As-1)/2)';  % Reference signal 

%% INL
[INL] = readINL("MeasuredINL/measurements_2023-09-10.csv", nob)';  % Read INL from file 
INL = dictionary(Q, INL); % Dictionary associate INL corresponding to the quantization levels

%% Direct Qunatization
u_direct= directQuantization(Q, ref);  % Direct Quantization
u_direct_INL = addINL(u_direct, INL);


%% MPC Setup
N = 10;
x0 = [0,0,0]';
% Get MPC Control 
% Provides optimal quantizer levels  at each prediction horizon 
getControl = getControlMPC(N, Q, A, B, C, D, 'gurobi');

%% Unifrom MPC
tstart = cputime; 
uni_mpc = [];  % container to store optimal uniform 
                                 % quantizer value values
% Moving horizon implementation 
for k = 1:length(ref)-N

    ref_in = ref(k:k+N-1);          % Reference value at prediction time 
                                    % intance t = k to t = k+N
    inputs = {x0, [ref_in']};       % Inputs: Initial state and reference 
                                    % value at t = k
    u_k = getControl(inputs);           % Optimal quantization levels at t = k
    x_new = A*x0 + B*(u_k(1)-ref(k));   % State prediction using obtained 
                                        % optimal value
    x0 = x_new;                         % Set predicted state as initial 
                                        % condition for optimization over 
                                        % next prediction horizon 
    uni_mpc =[uni_mpc; u_k(1)];                % Store applied  control
end
tend = cputime - tstart


%% Add INL to optimal uniform quantization post quantization
% This is equivalent to applying the unifrom quantized signal to the real
% DAC (i.e., the DAC with the INL)
uni_mpc_wINL = addINL(uni_mpc, INL);  % Add INL to the obtained optimal 
                                      % unifrom quantization levels post  quantization. 
%% Uniform MPC with INL
x0 = [0;0;0];       % Inital state
tic
uni_mpc_INL = []; % container to store value   
uni_mpc_INL_trun = []; % container to store value before 
                       % adding INL to the respective quantization levels.
% MHOQ loop 
for k = 1:length(ref)-N
    ref_in = ref(k:k+N-1);      % Reference value at prediction time 
                                % intance t = k to t = k+N
    inputs = {x0, [ref_in']};   % Inputs: Initial state and reference 
                                % value at t = k
    u_k = getControl(inputs);   % Optimal quantization levels at t = k

    % Add INL 
    u_act = addINL(u_k, INL);   % Add INL to the corresponding levels to 
    x_new = A*x0 + B*(u_act(1)-ref(k)); % Apply the updated value for the 
                                        % state prediction 
    x0 = x_new;                         % Set predicted state as initial 
                                        % condition for optimization over 
                                        % next prediction horizon  
    uni_mpc_INL = [uni_mpc_INL;  u_act(1)];          % Store applied  control (quantization level)
    uni_mpc_INL_trun = [uni_mpc_INL_trun;  u_k(1)];  % Store control (quantization level) before adding INL
                                        % Required for streaming in the
                                        % real DAC
end
toc

%% Signal filter
sl = length(uni_mpc);
filt_ref = filter(b,a, ref);
filt_direct = filter(b,a,u_direct);
filt_direct_INL = filter(b,a,u_direct_INL);
filt_uni_mpc= filter(b,a, uni_mpc(1:sl));
filt_uni_mpc_wINL= filter(b,a, uni_mpc_wINL(1:sl));
filt_uni_mpc_INL= filter(b,a, uni_mpc_INL(1:sl));

%% variance 
var_direct = var(filt_ref-filt_direct);
var_direct_INL = var(filt_ref-filt_direct_INL);
var_uni_mpc = var(filt_ref(1:sl)-filt_uni_mpc);
var_uni_mpc_wINL = var(filt_ref(1:sl)-filt_uni_mpc_wINL);
var_uni_mpc_INL = var(filt_ref(1:sl)-filt_uni_mpc_INL);


%%
figure
x_ax = categorical({'Direct','Direct-INL','MPC','MPC2-INL','MPC-INL'});
x_ax = reordercats(x_ax,{'Direct','Direct-INL','MPC','MPC2-INL','MPC-INL'} );
y_ax = [var_direct,var_direct_INL, var_uni_mpc, var_uni_mpc_wINL, var_uni_mpc_INL];
b1 = bar(x_ax, y_ax);
xtips2 = b1(1).XEndPoints;
ytips2 = b1(1).YEndPoints;
labels2 = string(b1(1).YData);
text(xtips2,ytips2,labels2,'HorizontalAlignment','center','VerticalAlignment','bottom')

%% SINAD
sinad_inl_mpc = sinad(filt_uni_mpc_INL);
enob = (sinad_inl_mpc - 1.76)/6.02;


