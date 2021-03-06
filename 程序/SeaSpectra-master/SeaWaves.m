clc;
clear all;
% close all;

%%
% Input data 
U10 = 8;   %Wind speed at 10m in m/s
U19half = 1.026 * U10;

g=9.814;     %Gravitational acceleration  m/s^2

Hs = 0.22 * (U10 ^ 2) / g;
Tm = (2 * pi * U19half) / (0.877 * g);  %Peak period (s)    wp = 0.877g / U19.5 so we solve for Tm with w = 2pi * f = 2pi/T
% Hs=12;      % significant wave height (m)
%Tm=10*(0.327*exp(-0.315*3.3)+1.17);        % Peak period (s)                       Tm=Tp=T0
% Tz=10;      % Zero-crossing period (s)
Omega=0.01:0.01:5;
Cap = 5;                          % This is the maximum considered Omega
TEnd=100;

% Type 1 is Jonswap and type 2 is Pierson-Moskowitz
[S, Amp, t] = SeaSpectrum('Omega', Omega ,'Hs', Hs, 'Tm' ,Tm, 'Type', 1, 'TEnd', TEnd, 'Cap', Cap, 'PlotSpectrum', 1, 'U10', U10);

%%
% Generating the signal using the first method
OmegaGap = Omega(2) - Omega(1);
rng(1)  %Setting the eed number for the random number generator
PhaseDiff = 2 * pi * rand(1, length(Omega));
Signal1 = sum(Amp' .* cos(Omega' .* t + PhaseDiff'));

%Generating the signal using the second method
rng(1)
Cn = randn(1, length(Omega));
Dn = randn(1, length(Omega));
AmpA = (S .* OmegaGap) .^ 0.5 .* Cn;
AmpB = (S .* OmegaGap) .^ 0.5 .* Dn;
Signal2 = sum(AmpA' .* cos(Omega' .* t) + AmpB' .* sin(Omega' .* t));

%%
figure
YRange = max(max(abs(Signal1)),max(abs(Signal2)));
subplot(4, 1, 1)
hold on 
plot(Omega, S)
xlabel('Omega (rad/s)');
ylabel('Spectrum (m^2.s)');
xlim([0, Cap])
grid;

subplot(4, 1, 2)
plot(Omega, Amp)
xlabel('Omega (rad/s)');
ylabel('Amplitude (m)');
xlim([0, Cap])
grid;

subplot(4, 1, 3)
% area(t, Signal1);
plot(t, Signal1)
xlabel('time (s)');
ylabel('Magnitude (m)');
title('Signal1 = sum(Amp.* cos(Omega .* t + PhaseDiff)')
grid;
ylim(1.2 * [-YRange, YRange])

subplot(4,1,4)
% area(t, Signal2)
plot(t,Signal2)
xlabel('time (s)');
ylabel('Magnitude (m)');
title('Signal2=sum(AmpA .* cos(Omega.*t)+AmpB .* sin(Omega.*t)')
ylim(1.2 * [-YRange, YRange])
grid;