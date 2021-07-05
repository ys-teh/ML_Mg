close all; clear; clc;

%% occopt 4

% kpoint 6 6 6
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy6 = [-3.59855803581636 -3.59849121245208 -3.59850683686567 -3.59862643660382 -3.59877742990628];
total_energy6 = [-3.59860407009533 -3.59870020064970 -3.59878468716049 -3.59888005204754 -3.59908185698039];
% entropy_energy6 = total_energy - internal_energy;
% estimated_energy6 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy6,'*-');
hold on;

% kpoint 8 8 8
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy8 = [-3.60037994022182 -3.60036494599473 -3.60036681166893 -3.60033861515251 -3.60059121236121];
total_energy8 = [-3.60037644250581 -3.60038893494704 -3.60039727788525 -3.60041607655823 -3.60042864296485];
% entropy_energy8 = total_energy - internal_energy;
% estimated_energy8 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy8,'*-');
hold on;

% kpoint 10 10 10
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy10 = [-3.59966896636719 -3.59964590377360 -3.59960329483995 -3.59954605463334 -3.59969622217078];
total_energy10 = [-3.59968016900411 -3.59970870909207 -3.59974088161849 -3.59981175102202 -3.60006370765235];
% entropy_energy10 = total_energy - internal_energy;
% estimated_energy10 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy10,'*-');
hold on;

% kpoint 12 12 12
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy12 = [-3.60008311460677 -3.60007237351409 -3.60006591046905 -3.60007003867010 -3.59987826361402];
total_energy12 = [-3.60006597091781 -3.60003846195807 -3.60002713558873 -3.60000788671802 -3.60003293687549];
% entropy_energy12 = total_energy - internal_energy;
% estimated_energy12 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy12,'*-');
hold on;

% kpoint 14 14 14
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy14 = [-3.60016358042161 -3.60018830273318 -3.60020186481734 -3.60021712941023 -3.60024723713220];
total_energy14 = [-3.60018044495071 -3.60020222857792 -3.60020502710575 -3.60020339447681 -3.60017801491747];
% entropy_energy14 = total_energy - internal_energy;
% estimated_energy14 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy14,'*-');
hold on;

% kpoint 16 16 16
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy16 = [-3.59989478596715 -3.59986821060205 -3.59985380132933 -3.59981177922167 -3.59990802811497];
total_energy16 = [-3.59989220406463 -3.59990127917600 -3.59991677515257 -3.59995123662891 -3.60008845725836];
% entropy_energy16 = total_energy - internal_energy;
% estimated_energy16 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy16,'*-');
hold on;

% kpoint 18 18 18
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy18 = [-3.60007283835717 -3.60006774452801 -3.60004747553691 -3.60001676330661 -3.60003220063402];
total_energy18 = [-3.60006697562328 -3.60005342614030 -3.60005119785571 -3.60005877575990 -3.60011497189716];
% entropy_energy18 = total_energy - internal_energy;
% estimated_energy18 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy18,'o-');
hold on;

% kpoint 20 20 20
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy20 = [-3.60005865910930 -3.60004921845977 -3.60005137737589 -3.60004175354243 -3.60003960457381];
total_energy20 = [-3.60005606706621 -3.60005852473126 -3.60006232903781 -3.60006723332924 -3.60011638673219];
% entropy_energy20 = total_energy - internal_energy;
% estimated_energy20 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy20,'o-');
hold on;

% % kpoint 22 22 22
% tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
% internal_energy22 = [-3.59936884330891 -3.59936771441100 -3.59928623928588 -3.59895912415178 -3.59897682905389];
% total_energy22 = [-3.59932479715861 -3.59925468276943 -3.59921846521121 -3.59924932461164 -3.59965974445987];
% % entropy_energy22 = total_energy - internal_energy;
% % estimated_energy22 = 0.5*(internal_energy+total_energy);
% plot(tsmear,total_energy22,'o-');
% hold on;

% kpoint 30 30 30
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy30 = [-3.60004746374317 -3.60004900868315 -3.60004065281685 -3.60001412492024 -3.60001937828963];
total_energy30 = [-3.60004768635069 -3.60004590224947 -3.60004591542214 -3.60005269174396 -3.60011284960509];
% entropy_energy30 = total_energy - internal_energy;
% estimated_energy30 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy30,'o-');
hold on;

legend('6','8','10','12','14','16','18','20','30');

figure(2);
kpoint = [6 8 10 12 14 16 18 20];
matrix = [abs(total_energy6 - total_energy30);
    abs(total_energy8 - total_energy30);
    abs(total_energy10 - total_energy30);
    abs(total_energy12 - total_energy30);
    abs(total_energy14 - total_energy30);
    abs(total_energy16 - total_energy30);
    abs(total_energy18 - total_energy30);
    abs(total_energy20 - total_energy30)];
semilogy(kpoint,matrix(:,1),'*-');
hold on;
semilogy(kpoint,matrix(:,2),'*-');
hold on;
semilogy(kpoint,matrix(:,3),'*-');
hold on;
semilogy(kpoint,matrix(:,4),'*-');
hold on;
semilogy(kpoint,matrix(:,5),'*-');
l = legend('smear = 0.002','smear = 0.005','smear = 0.007','smear = 0.01','smear = 0.02');
l.Location = 'Southwest';
xlabel('kpoint grid');
ylabel('Total free energy error (Hartree)');
grid on;