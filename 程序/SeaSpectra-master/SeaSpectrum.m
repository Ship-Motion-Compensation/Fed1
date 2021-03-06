function [S, Amp, t] = SeaSpectrum(varargin)
    % This is a function to create a Jonswap/Pierson-Moskowitz spectrum

    %%
    % Parse the required data
    p = inputParser;
    %Required   
    addParameter(p,'Omega',@isrow); 
    addParameter(p,'Hs',@isnumeric);% significant wave height
    addParameter(p,'Tm',0);         % Peak period (s)
    addParameter(p,'Tz',0);
    %Optional
    addParameter(p,'Type',1); %type 1 is Jonswapis and type 2  Pierson-Moskowitz
    addParameter(p,'Cap',2); %Getting rid of the end of the spectrum 
    addParameter(p,'TEnd',3600*3/15);  % The duration of a the signal
    addParameter(p,'PlotSpectrum',1);  % if this one is set to 1, nothing happens, if 2 it will draw the spectrum
    addParameter(p, 'U10', 20);
    
    parse(p,varargin{:});   

    Omega = p.Results.Omega;
    Hs = p.Results.Hs;
    Tm = p.Results.Tm;
    Tz = p.Results.Tz;
    Type = p.Results.Type;   
    TEnd = p.Results.TEnd;  
    Cap = p.Results.Cap; 
    PlotSpectrum = p.Results.PlotSpectrum;
    U10 = p.Results.U10;

    % Default value
    g=9.814;     %Gravitational acceleration  m/s^2
    %g=32.74    %Gravitational acceleration  ft/s^2

%     U10 = 20;   %Wind speed at 10m in m/s
    U19half = 1.026 * U10;
    Omega0 = g / U19half;
    
    if Type == 1
        disp('Jonswap Spectrum')
        Gamma = 3.3;
        Beta = 5 / 4;
        F = 30000;    %Fetch - distance from the Lee shore
%         alphabar = 5.058 * (1 - 0.287 * log(Gamma)) * (Hs / Tm^2)^2;    % Modified Phillips constant
        alphaJ = 0.076 * (((U10 ^ 2) / (F * g)) ^ 0.22);
        alpha = alphaJ;
%         Omegam = 2 * pi / Tm
        Omegam = 22 * (((g ^ 2) / (U10 * F)) ^ (1 / 3));
    elseif Type == 2
        disp('Pierson-Moskowitz Spectrum')
        Gamma = 1;
        Beta = 0.74;
        alpha = 0.0081;
        Omegam = Omega0;
    end

    if Tz ~= 0
        Tm= Tz*(0.327*exp(-0.315*Gamma)+1.17);
        % Tm= Tz*((11+Gamma)./(5+Gamma)).^.5
        disp('Using Tz')
    end

    
%     Beta = 0.74;
    SigmaA = 0.07;
    SigmaB = 0.09;
    format short

    %%
    OmegaGap = Omega(2) - Omega(1); 
    sigma = (Omega <= Omegam) * SigmaA + (Omega > Omegam) * SigmaB;
%     A = exp(-((Omega / Omegam - 1) ./ (sigma * sqrt(2))) .^ 2);
    A = exp(-((Omega - Omegam) .^ 2) ./ (2 * (sigma .^ 2) * (Omegam ^ 2)));
    fprintf('U10= %d, alpha= %d, Omega0 = %d, Omegam = %d\n',U10,alpha, Omega0, Omegam)
    S = alpha * g^2 .* Omega .^ -5 .* exp(-(Beta * (Omega / Omegam) .^ -4)) .* (Gamma .^ A);
    S(Omega > Cap) = 0;
    Amp = (2 * S .* OmegaGap) .^ 0.5;
    t = linspace(0, TEnd, length(Omega));

    %%
    if PlotSpectrum == 2
        figure
        plot(Omega, S)
        ylabel('Spectrum (m/s')
        xlabel('Omega (rad/s)');ylabel('Spectrum (m^2.s)');
        grid;
        if Type==1
            ti1=('JONSWAP Spectrum, ');
        elseif Type==2
            ti1=('Pierson-Moskowitz Spectrum, ');
        end
        if Tz==0    %there are two methods for converting Tz to Tm
            ti2=sprintf('Tm=%d, ', Tm);   
        else
           ti2=sprintf('Tz=%d, ', Tz); 
        end
        ti3 = sprintf('Hs=%d', Hs);
        title([ti1,ti2,ti3]);
    end
        
    fprintf('The default values: \n g=%d \n Gamma=%d \n Beta=%d \n SigmaA=%d \n SigmaB=%d \n TEnd=%d, \n Cap=%d \n', g, Gamma, Beta, SigmaA, SigmaB, TEnd, Cap )
            