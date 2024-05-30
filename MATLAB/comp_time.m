% 
N2 = [76.24, 82.07, 82.95, 66.16]./(8*);
N3 = [119.7, 130.45, 131.45, 137];
N4 = [181.24, 234.5, 249, 248.5];
N5 = [323.77, 522.07, 578.72, 534];
N10 = [3.06e3, 6.16e3, 6.63e3, 6.58e3];
bits = [4, 8, 12, 16];
% figure
% plot(bits, N2)
% hold on 
% plot(bits, N3)
% hold on 
% plot(bits, N4)
% hold on 
% plot(bits, N5)
% hold on 
% plot(bits, N10)
% xlabel('Bits')
% ylabel('Computation Time (sec)')
% legend('N = 2','N = 3','N =4','N = 5','N = 10')

figure
semilogy(bits, N2)
hold on 
semilogy(bits, N3)
hold on 
semilogy(bits, N4)
hold on 
semilogy(bits, N5)
hold on 
semilogy(bits, N10)
xlabel('Bits')
ylabel('Computation Time (sec)')
legend('N = 2','N = 3','N =4','N = 5','N = 10')
%%
B4 = [76.29, 119.7, 181.24, 323.77, 3.06e3];
B8 = [82.07, 130.45, 234.5, 522.07,6.16e3];
B12 = [82.95, 131.45, 249, 578.72,6.63e3 ];
B16 = [66.16, 137, 248.5, 534, 6.58e3];
N =[1,2,3,4,5];
% N = ['2','3','4','5','10'];
% figure 
% plot(N, B4)
% hold on 
% plot(N, B8)
% hold on 
% plot(N, B12)
% hold on 
% plot(N, B16)
% xlabel('Prediction horizon (N)')
% ylabel('Computation Time (sec)')
% legend('4 bits','8 bits','12 bits','16 bits')

figure 
semilogy(N, B4)
hold on 
semilogy(N, B8)
hold on 
semilogy(N, B12)
hold on 
semilogy(N, B16)
xlabel('Prediction horizon (N)')
ylabel('Computation Time (sec)')
legend('4 bits','8 bits','12 bits','16 bits')