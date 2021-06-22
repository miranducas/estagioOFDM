
clear all
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generating channels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_channels = 33000;
L_veh = 21;
L_ped = 11;
v1 = 4 * 1e3 / 3600; % Mobile speed (m/s)
v2 = 100 * 1e3 / 3600; % Mobile speed (m/s)
fc = 2e+9; % Carrier frequency
c = physconst('LightSpeed'); % Speed of light in free space
max_doppler_shift1 = v1*fc/c; 
max_doppler_shift2 = v2*fc/c; 
Ts = 200e-9;

% y_veh = zeros(num_channels,L_veh);
% 
% for i=1:num_channels
% ch_resp_veh = stdchan(Ts,max_doppler_shift2,'itur3GVAx');    
% ch_resp_veh.StoreHistory=1;
% x=zeros(1,L_veh);
% x(1)=1;
% y_veh(i,:)=filter(ch_resp_veh,x);
% end

% y_ped = zeros(num_channels,L_ped);
% for i=1:num_channels
% ch_resp_ped = stdchan(Ts,max_doppler_shift1,'itur3GPAx');
% ch_resp_ped.StoreHistory=1;
% x=zeros(1,L_ped);
% x(1)=1;
% y_ped(i,:)=filter(ch_resp_ped,x);
% end


L_vehped = 21;
y_vehped = zeros(num_channels,L_vehped);
channels_met = num_channels/2;
for i=1:2:num_channels-1
    
ch_resp_ped = stdchan(Ts,max_doppler_shift1,'itur3GPAx');
ch_resp_ped.StoreHistory=1;

ch_resp_veh = stdchan(Ts,max_doppler_shift2,'itur3GVAx');    
ch_resp_veh.StoreHistory=1;

x=zeros(1,L_vehped);
x(1)=1;

y_vehped(i,:)=filter(ch_resp_ped,x);
y_vehped(i+1,:)=filter(ch_resp_veh,x);

end


% 
% y_ped_main=y_ped(1:5000,:); 
% y_ped_train = y_ped(5001:25000,:);
% y_ped_test=y_ped(25001:33000,:);
% save('channel_ped.mat','y_ped_main','y_ped_train','y_ped_test');
% 
% y_veh_main=y_veh(1:5000,:); 
% y_veh_train=y_veh(5001:25000,:);
% y_veh_test=y_veh(25001:33000,:);
% save('channel_veh.mat','y_veh_main','y_veh_train','y_veh_test');
% 
y_vehped_main=y_vehped(1:5000,:);
y_vehped_train=y_vehped(5001:25000,:);
y_vehped_test=y_vehped(25001:33000,:);
save('channel_vehped.mat','y_vehped_main','y_vehped_train','y_vehped_test');



 %figure,
 %stem(abs(y))
% figure,
% load('vehA200channel.mat')
% stem(abs(vehA200channel(3,:)))
% 
% figure,
% stem(ch_resp.PathDelays)
