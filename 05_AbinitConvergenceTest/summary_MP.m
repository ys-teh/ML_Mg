close all; clear; clc;

%% occopt 6

% % kpoint 8 8 8
% tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
% internal_energy = [-3.60038086787643 -3.60038220903460 -3.60038195532852 -3.60037809673894 -3.60041756226248 -3.60030866389924 -3.60090185242645];
% total_energy = [-3.60038103656585 -3.60038127681367 -3.60038037982845 -3.60037907508159 -3.60036810615999 -3.60035482323657 -3.60012578937789];
% entropy_energy = total_energy - internal_energy;
% estimated_energy = 0.5*(internal_energy+total_energy);
% 
% plot(tsmear,total_energy,'*-');
% hold on;
% %plot(tsmear(1:3),estimated_energy(1:3),'<-');
% hold on;
% 
% % kpoint 10 10 10
% tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
% internal_energy = [-3.59968120049443 -3.59968087012338 -3.59967753522146 -3.59967462386689 -3.59966388928386 -3.59966245318159 -3.60018379652100];
% total_energy = [-3.59968150424211 -3.59968173835648 -3.59968401299126 -3.59969144684819 -3.59972395680457 -3.59978516534358 -3.59987919204766];
% entropy_energy = total_energy - internal_energy;
% estimated_energy = 0.5*(internal_energy+total_energy);
% 
% plot(tsmear,total_energy,'*-');
% hold on;
% %plot(tsmear(1:3),estimated_energy(1:3),'<-');
% hold on;
% 
% % % kpoint 12 12 12
% % tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
% % internal_energy = [];
% % total_energy = [];
% % entropy_energy = total_energy - internal_energy;
% % estimated_energy = 0.5*(internal_energy+total_energy);
% % 
% % plot(tsmear,total_energy,'*-');
% % hold on;
% % %plot(tsmear(1:3),estimated_energy(1:3),'<-');
% % hold on;
% 
% % kpoint 20 20 20
% tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
% internal_energy = [-3.60006089240605 -3.60006139656401 -3.60006239445033 -3.60005829613720 -3.60006174680470 -3.60006242167302 -3.60032928921104];
% total_energy = [-3.60006089240605 -3.60006139656401 -3.60005931101595 -3.60005744992582 -3.60005635596903 -3.60004895033641 -3.59990678280719];
% entropy_energy = total_energy - internal_energy;
% estimated_energy = 0.5*(internal_energy+total_energy);
% 
% plot(tsmear,total_energy,'*-');
% hold on;
% %plot(tsmear(1:3),estimated_energy(1:3),'<-');
% hold on;
% 
% legend('8','10','12','20');

% kpoint 6 6 6
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy6 = [-3.59854971401350 -3.59850097980987 -3.59860535103123 -3.59873695239540 -3.59911840245441];
total_energy6 = [-3.59862550893121 -3.59877269179844 -3.59886631005385 -3.59894773626247 -3.59901989758942];
% entropy_energy6 = total_energy - internal_energy;
% estimated_energy6 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy6,'*-');
hold on;

% kpoint 8 8 8
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy8 = [-3.60037809673898 -3.60041756226264 -3.60038348723612 -3.60030866389899 -3.60061273475012];
total_energy8 = [-3.60037907508156 -3.60036810615996 -3.60035294443590 -3.60035482323654 -3.60034731050764];
% entropy_energy8 = total_energy - internal_energy;
% estimated_energy8 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy8,'*-');
hold on;

% kpoint 10 10 10
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy10 = [-3.59967462386697 -3.59966388928385 -3.59966163112413 -3.59966245318231 -3.59977245968271];
total_energy10 = [-3.59969144684825 -3.59972395680460 -3.59974872543279 -3.59978516534362 -3.59990754961456];
% entropy_energy10 = total_energy - internal_energy;
% estimated_energy10 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy10,'*-');
hold on;

% kpoint 12 12 12
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy12 = [-3.60007907101543 -3.60009217436918 -3.60008763522711 -3.60008101135818 -3.60010247030787];
total_energy12 = [-3.60007437141652 -3.60005821710690 -3.60004549776303 -3.60002816643038 -3.59998922418184];
% entropy_energy12 = total_energy - internal_energy;
% estimated_energy12 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy12,'*-');
hold on;

% kpoint 14 14 14
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy14 = [-3.60016288270747 -3.60017608877590 -3.60020543304985 -3.60027177177574 -3.60036136111280];
total_energy14 = [-3.60017804441550 -3.60020176925552 -3.60020741501255 -3.60019588671196 -3.60007191195964];
% entropy_energy14 = total_energy - internal_energy;
% estimated_energy14 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy14,'*-');
hold on;

% kpoint 16 16 16
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy16 = [-3.59989522662721 -3.59987682081364 -3.59985549569573 -3.59983406227156 -3.60004250572409];
total_energy16 = [-3.59989331572694 -3.59989757380019 -3.59990959117836 -3.59993766668080 -3.59999518217068];
% entropy_energy16 = total_energy - internal_energy;
% estimated_energy16 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy16,'*-');
hold on;

% kpoint 18 18 18
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy18 = [-3.60007175331720 -3.60006410299610 -3.60004502349931 -3.60003161099756 -3.60021226690646];
total_energy18 = [-3.60006486112717 -3.60005457038498 -3.60005383635245 -3.60006194358017 -3.60003136888422];
% entropy_energy18 = total_energy - internal_energy;
% estimated_energy18 = 0.5*(internal_energy+total_energy);
plot(tsmear,total_energy18,'o-');
hold on;

% kpoint 20 20 20
tsmear = [2e-3 5e-3 7e-3 1e-2 2e-2];
internal_energy20 = [-3.60005829613714 -3.60006174680475 -3.60006464153982 -3.60006242167312 -3.60016303313243];
total_energy20 = [-3.60005744992582 -3.60005635596912 -3.60005375404778 -3.60004895033640 -3.60002127384603];
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
internal_energy30 = [-3.60004891730246 -3.60005062923023 -3.60004387395751 -3.60003285429757 -3.60016667958441];
total_energy30 = [-3.60004718662695 -3.60004294228174 -3.60004091671256 -3.60004200480545 -3.60002231651294];
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