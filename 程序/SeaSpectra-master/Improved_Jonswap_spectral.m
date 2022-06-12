clc
clear;
y=[];
g=9.8;
A=0.0081*g^2;

U1=15;
x = 0.1:0.01:2;
j=length(x);
B1=0.74*((g/U1)^4);
U2=12;
B2=0.74*((g/U2)^4);
U3=10;
B3=0.74*((g/U3)^4);
for i=1:j
y1(i)=(A/x(:i)^5)*exp(-B1/x(:i)^4);
y2(i)=(A/x(:i)^5)*exp(-B2/x(:i)^4);
y3(i)=(A/x(:i)^5)*exp(-B3/x(:i)^4);
end

plot(xy1xy2xy3);