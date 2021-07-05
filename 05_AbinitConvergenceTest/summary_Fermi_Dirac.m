%close all; clear; clc;

%% occopt 3

% %tsmear 0.001
% kpoint = [6 8 10 12 14];
% internal_energy = [-3.59848551081973 -3.60031113913207 -3.59959980679520 -3.60001131819173 -3.60010202040313];
% total_energy = [-3.5988259287 -3.6004483307 -3.5998001401 -3.6001402749 -3.6002630237];
% 
% scatter(kpoint,internal_energy);
% hold on;
% scatter(kpoint,total_energy);

% %tsmear 0.005
% kpoint = [6 8 10 12 14];
% internal_energy = [-3.59697563784254 -3.59853148481066 -3.59784427532290 -3.59813938800640 -3.59833835552651];
% total_energy = [-3.60098427662988 -3.60220296987386 -3.60178982724212 -3.60190109968975 -3.60199738305048];
% estimated_energy = 0.5*(internal_energy+total_energy);
% 
% scatter(kpoint,internal_energy);
% hold on;
% scatter(kpoint,estimated_energy);

%% occopt 3

% kpoint 6 6 6
tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
internal_energy = [-3.59857026836915 -3.59853640735226 -3.59848551081973 -3.59837286658797 -3.59697563784254 -3.59109835956858 -3.52977915003191];
total_energy = [-3.59858544246396 -3.59867125794758 -3.59882592868842 -3.59920542978182 -3.60098427662988 -3.60693389660479 -3.66960413629977];
entropy_energy = total_energy - internal_energy;
estimated_energy = 0.5*(internal_energy+total_energy);

plot(tsmear,total_energy,'*-');
hold on;
%plot(tsmear(1:3),estimated_energy(1:3),'<-');
hold on;

% kpoint 8 8 8
tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
internal_energy = [-3.60038082356739 -3.60036315836355 -3.60031113913207 -3.60008143801846 -3.59853148481065 -3.59269818796253 -3.53020143403574];
total_energy = [-3.60038192902765 -3.60039695547015 -3.60044833072282 -3.60066014599658 -3.60220296987385 -3.60778163154018 -3.66969237014358];
entropy_energy = total_energy - internal_energy;
estimated_energy = 0.5*(internal_energy+total_energy);

plot(tsmear,total_energy,'*-');
hold on;
%plot(tsmear(1:3),estimated_energy(1:3),'<-');
hold on;

% kpoint 10 10 10
tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
internal_energy = [-3.59967997750079 -3.59965894901561 -3.59959980679519 -3.59937391923955 -3.59784427532290 -3.59219038353383 -3.53015875921485];
total_energy = [-3.59968404949309 -3.59971978689196 -3.59980014011325 -3.60007681000873 -3.60178982724213 -3.60759077420232 -3.66968518261863];
entropy_energy = total_energy - internal_energy;
estimated_energy = 0.5*(internal_energy+total_energy);

plot(tsmear,total_energy,'*-');
hold on;
%plot(tsmear(1:3),estimated_energy(1:3),'<-');
hold on;

% kpoint 12 12 12
tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
internal_energy = [-3.60007185380157 -3.60006061318182 -3.60001131819173 -3.59977917565307 -3.59813938800640 -3.59227604291274 -3.53016064112989];
total_energy = [-3.60007481612057 -3.60009264176085 -3.60014027493762 -3.60034506267394 -3.60190109968975 -3.60760884746863 -3.66968551281202];
entropy_energy = total_energy - internal_energy;
estimated_energy = 0.5*(internal_energy+total_energy);

plot(tsmear,total_energy,'*-');
hold on;
%plot(tsmear(1:3),estimated_energy(1:3),'<-');
hold on;

% kpoint 20 20 20
tsmear = [1e-4 5e-4 1e-3 2e-3 5e-3 1e-2 3e-2];
internal_energy = [-3.60006043628547 -3.60004115178380 -3.59998398568352 -3.59975920104008 -3.59816496779757 -3.59231518633469 -3.53016099741866];
total_energy = [-3.60006133718379 -3.60007648278132 -3.60013124489052 -3.60035313283901 -3.60192590946590 -3.60762133366637 -3.66968553990843];
entropy_energy = total_energy - internal_energy;
estimated_energy = 0.5*(internal_energy+total_energy);

plot(tsmear,total_energy,'*-');
hold on;
%plot(tsmear(1:3),estimated_energy(1:3),'<-');

xlabel('Smearing parameter (Hartree)');
ylabel('Total free energy (Hartree)');
legend('6x6x6','8x8x8','10x10x10','12x12x12','20x20x20');

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


%% occopt 3
figure(2);
internal_energy6 = [-3.59837286658797 -3.59697563784254 -3.59109835956858 -3.52977915003191];
total_energy6 = [-3.59920542978182 -3.60098427662988 -3.60693389660479 -3.66960413629977];
internal_energy8 = [-3.60008143801846 -3.59853148481065 -3.59269818796253 -3.53020143403574];
total_energy8 = [-3.60066014599658 -3.60220296987385 -3.60778163154018 -3.66969237014358];
internal_energy10 = [-3.59937391923955 -3.59784427532290 -3.59219038353383 -3.53015875921485];
total_energy10 = [-3.60007681000873 -3.60178982724213 -3.60759077420232 -3.66968518261863];
internal_energy12 = [-3.59977917565307 -3.59813938800640 -3.59227604291274 -3.53016064112989];
total_energy12 = [-3.60034506267394 -3.60190109968975 -3.60760884746863 -3.66968551281202];
internal_energy14 = [-3.59991192018554 -3.59833835552652 -3.59237792018002 -3.53016114707412];
total_energy14 = [-3.60048586793464 -3.60199738305048 -3.60763564368595 -3.66968555226134];
internal_energy16 = [-3.59957020300005 -3.59803249340258 -3.59228159357322 -3.53016096659053];
total_energy16 = [-3.60022822773972 -3.60187923771949 -3.60761453875450 -3.66968553755058];
internal_energy20 = [-3.59975920104008 -3.59816496779757 -3.59231518633469 -3.53016099741866];
total_energy20 = [-3.60035313283901 -3.60192590946590 -3.60762133366637 -3.66968553990843];
internal_energy24 = [-3.59976438827715 -3.59817313261287 -3.59231731794972 -3.53016099755200];
total_energy24 = [-3.60035785454596 -3.60192911375112 -3.60762168623515 -3.66968553995186];
internal_energy30 = [-3.59974349952228 -3.59815711174473 -3.59231532018947 -3.53016099747418];
total_energy30 = [-3.60034354353764 -3.60192443625309 -3.60762140230361 -3.66968553995771];

kpoint = [6 8 10 12 14 16 20 24];
matrix = [abs(total_energy6 - total_energy30);
    abs(total_energy8 - total_energy30);
    abs(total_energy10 - total_energy30);
    abs(total_energy12 - total_energy30);
    abs(total_energy14 - total_energy30);
    abs(total_energy16 - total_energy30);
    abs(total_energy20 - total_energy30);
    abs(total_energy24 - total_energy30)];
semilogy(kpoint,matrix(:,1),'*-');
hold on;
semilogy(kpoint,matrix(:,2),'*-');
hold on;
semilogy(kpoint,matrix(:,3),'*-');
hold on;
semilogy(kpoint,matrix(:,4),'*-');
l = legend('smear = 0.002','smear = 0.005','smear = 0.01','smear = 0.03');
l.Location = 'Southwest';
xlabel('kpoint grid');
ylabel('Total free energy error (Hartree)');
grid on;