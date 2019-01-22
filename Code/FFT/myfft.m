
% infile - input file
% outfile - output file
% Fs - frequency of sampling
% n - order of FFT

X = dlmread(infile); X = X - mean(X);
% L = length(X); Fs = L;
% n = 2^nextpow2(L);
Y = fft(X, n);
f = Fs*(0:(n/2))/n;
P = abs(Y/n);
P = P(1:n/2+1);
outM = [f' P];
dlmwrite(outfile, outM);
