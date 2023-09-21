% @ SPL 석사과정생 권민성 
% 엑셀파일로부터 Figure을 Plot하는 code
% 

clc, clear
format long

filename = 'standford_rd.xlsx'; % 엑셀 파일의 위치를 Access 
xl = xlsread(filename,1);
%%
singleres = xl(1:3, 1:2:6);
awmr_pool = xl(12:14, 1:end);
awmr_pool_v2 = xl(28:30,1:end);
awmr_min = xl(41:43, 1:end);
octree_pool = xl(20:22, 1:end);
octree_pool_v2 = xl(36:38,1:end);
octree_min = xl(46:48, 1:end);
%%
figure(3)
plot(singleres(1,:), singleres(2,:),'-o', LineWidth = 2)
hold on;
plot(awmr_pool(1,:), awmr_pool(2,:),'-o', LineWidth = 2);
plot(octree_pool(1,:), octree_pool(2,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','AWMR', 'Octree')
title('dataset: dragon, GT: 원본 메시')
%%
figure(4)
plot(singleres(1,:), singleres(3,:),'-o', LineWidth = 2)
hold on;
plot(awmr_pool(1,:), awmr_pool(3,:),'-o', LineWidth = 2);
plot(octree_pool(1,:), octree_pool(3,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','AWMR', 'Octree')
title('dataset: dragon, GT: voxelsize 0.001 singleres')

%%
figure(3)
plot(singleres(1,:), singleres(2,:),'-o', LineWidth = 2)
hold on;
% plot(awmr_pool(1,:), awmr_pool(2,:),'-ob', LineWidth = 2);
% plot(awmr_pool_v2(1,:), awmr_pool_v2(2,:),'-o', LineWidth = 2);
plot(awmr_min(1,:), awmr_min(2,:),'-o', LineWidth = 2);
plot(octree_min(1,:), octree_min(2,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','AWMR','Octree')
title('dataset: dragon, GT: 원본 메시')