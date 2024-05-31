function [deviation_in_lsb] =  readINL(filename, nob)
% Reads INL from the measuredINL file
INL = readmatrix(filename);
% Scaling according to nob
dac_levels = INL(:,1);
measured_voltage = INL(:,3);

% Scaled INL
measured_voltage = measured_voltage(1:2^nob);
dac_levels = dac_levels(1:2^nob);
ideal_voltage = linspace(measured_voltage(1),measured_voltage(end), length(dac_levels))';
deviation_in_v = ideal_voltage - measured_voltage;
v_per_lsb = (measured_voltage(end)- measured_voltage(1))/length(dac_levels);
deviation_in_lsb = deviation_in_v/v_per_lsb;
end