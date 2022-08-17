clc
clear all
%% Data Inputs 
Acc_EW = importdata('Nepal\ADIB.HHE.dat');
Acc_NS = importdata('Nepal\ADIB.HHN.dat');
Acc_ver = importdata('Nepal\ADIB.HHZ.dat');
Fs = 200;  
%% Signal Pre-Processing
%Filter Design
digfilt = designfilt('lowpassiir', 'PassbandFrequency', 20, 'StopbandFrequency', 25, 'PassbandRipple', 1, 'StopbandAttenuation', 60, 'SampleRate', 200);
% Filtering Data
Acc_EW_filt = filter(digfilt,Acc_EW);
Acc_NS_filt = filter(digfilt,Acc_NS);
Acc_ver_filt = filter(digfilt,Acc_ver);
Fhp = 0.8;  
[b1,a1] = butter(3,Fhp/Fs,'high'); % 
fildat = filter(b1,a1,Acc_ver); %  
vel = cumtrapz(fildat)./Fs; %  
[b2,a2] = butter(3,Fhp/Fs,'high'); 
fildat1 = filter(b2,a2,vel);  
dis = cumtrapz(fildat1)./Fs;   
peakToPeakRange = max(fildat) - min(fildat);
dt = 1/Fs;  
nt = length(fildat);  
time = (1:nt).*dt;  
%% Removing the delay introduced by lowpassiir filter
Acc_EW_dlycompensated = filtfilt(digfilt,Acc_EW);
Acc_NS_dlycompensated = filtfilt(digfilt,Acc_NS);
Acc_ver_dlycompensated = filtfilt(digfilt,Acc_ver);

%% SVM Algorithm gor P-Wave detection
stw = 1;     
ltw = 60;    
thresh = 4 ;  
thresh1 = 3;
%t = 1;      
nl = fix(ltw / dt);  
ns = fix(stw / dt);  
nt = length(fildat); 
sra = zeros(1, nt);
for k = nl+1:nt
    sta(k,1) = (1/ns)* trapz(abs(fildat(k-ns:k)));
    lta(k,1) = (1/nl)* trapz(abs(fildat(k-nl:k)));
 end
for l = nl+1: nt 
    sra(l) = sta(l)/lta(l);
end
  itm = find(sra > thresh);
    if ~isempty(itm)
      itmax = itm(1);
    end
    tp = itmax*dt;  
    fprintf('P-Wave detection time for threshold 4 =  %f second\n', tp);
    
  itm1 = find(sra > thresh1);
    if ~isempty(itm1)
      itmax1 = itm1(1);
    end
    tp1 = itmax1*dt; % P-wave arriving time 
    fprintf('P-Wave detection time for threshold 3 = %f second\n', tp1);
%% S-wave arrival time
pkHts = 0.72; % 10 percent
[pk2,t22] = findpeaks(Acc_NS_dlycompensated,Fs,'MinPeakHeight',pkHts*max(Acc_ver_dlycompensated),'Npeaks',1);
[pk3,t33] = findpeaks(Acc_EW_dlycompensated,Fs,'MinPeakHeight',pkHts*max(Acc_ver_dlycompensated),'Npeaks',1);

display(sprintf('S-wave found on EW component at %f seconds and on NS componet at %f seconds,', t33,t22));

if(t22<t33)
    display('S-wave detected first on North-South component');
else
    display('S-wave detected first on East-West component');
end
ts = min(t22,t33);
line([ts,ts],[min(get(gca,'Ylim'))],'linestyle','--','linewidth',2,'color','red');
    
%% Tauc , Pd and Magnitude calculations
vel_sq = vel.^2;
dis_sq = dis.^2;
r1 = trapz(vel_sq((itmax):(itmax+600)));
r2 = trapz(dis_sq((itmax):(itmax+600)));
r = r1/r2;
tauc = 2*pi/sqrt(r);
pd = max(dis((itmax):(itmax+600)));


%% Distance of earthquake from the seismometer
dist = (ts-tp)*8;
display(sprintf('Earthquake is estimated to be %f kilometers from the seismometer',dist))

%% Acceleration Plot
figure(1);
subplot(3,1,1)
plot(time,fildat,[tp tp],ylim,'r','LineWidth',1)
%plot(time,fildat)
title('Acceleration Data');
xlabel('Time (Sec)');
ylabel('Acceleration (cm/sec^2)');
grid on 
grid minor
%% Velocity Plot
subplot(3,1,2)
plot(time,vel)
title('Velocity Data');
xlabel('Time (Sec)');
ylabel('Velocity (cm/sec)');
grid on 
grid minor
%% Displacement Plot
subplot(3,1,3)
plot(time,dis)
title('Displacement Data');
xlabel('Time (Sec)');
ylabel('Displacement (cm)');
grid on 
grid minor

%% Plotting Spectogram of Original Signal and detecting the P-wave first arrival
figure(2)
box on
hold on
subplot(3,1,1)
plot(time,fildat,[tp tp],ylim,'r','LineWidth',2)
hold on
plot(time,fildat,[tp1 tp1],ylim,'g','LineWidth',2)
 
xlabel('Time (Sec)');
ylabel('Acceleration (cm/sec^2)');
grid on 
grid minor
axis tight

box on
s = spectrogram(abs(fildat),256,250,256,200,'yaxis');
subplot(3,1,2)
 
spectrogram(abs(fildat),256,250,256,200,'yaxis')
tp_in_min = tp/60;
tp_in_min1 = tp1/60;
%line([tp_in_min tp_in_min],[0,100],'Color','red','LineWidth',2);
line([tp_in_min1 tp_in_min1],[0,100],'Color','green','LineWidth',2);
grid on 
grid minor
axis tight;

box on
subplot(3,1,3)
thresh_spec = spectrogram(abs(fildat),256,250,256,200,'MinThreshold',-50,'yaxis');
thresh_spec1 = abs(thresh_spec);
spectrogram(abs(fildat),256,250,256,200,'MinThreshold',-50,'yaxis')
tp_in_min = tp/60;
tp_in_min1 = tp1/60;
line([tp_in_min1 tp_in_min1],[0,100],'Color','green','LineWidth',2);

grid on 
grid minor
axis tight;

Nr = 1;
Dr = [1,-1];
Ver_acc_dt = detrend(Acc_ver_dlycompensated);
NS_acc_dt = detrend(Acc_NS_dlycompensated);
EW_acc_dt = detrend(Acc_EW_dlycompensated);

vel_ver_nodrift = filter(Nr,Dr,Ver_acc_dt)/Fs; 
vel_NS_nodrift = filter(Nr,Dr,NS_acc_dt)/Fs;
vel_EW_nodrift = filter(Nr,Dr,EW_acc_dt)/Fs;

dis_ver = filter(Nr,Dr,vel_ver_nodrift)/Fs;
dis_NS = filter(Nr,Dr,vel_NS_nodrift)/Fs;
dis_EW = filter(Nr,Dr,vel_EW_nodrift)/Fs;

figure(3);
plot3(dis_NS,dis_EW,dis_ver);
grid on; view([-45,30]);
xlabel('N-S Direction in cm');
ylabel('E-W Direction in cm');
zlabel('Vertical Direction in cm');
title('Displacement of the seismometer in 3D');
set(gcf,'Name','Seismometer Trajectory');
set(gcf,'Units','Normalized');