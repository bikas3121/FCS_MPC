function [u_INL] = addINL(u, INL)
%% Inputs and Outputs
% Inputs:
    % u - quantization levels
    % INL - Dictionary with INL corresponding to each levels

% Output
    % u_INL - quantization levels with added INL 

%%  
u_INL = zeros(length(u),1);
for i = 1 :length(u)
    u_INL(i) = u(i) + INL(round(u(i)));
end
end