% @ SPL 석사과정생 권민성 
% 엑셀파일로부터 Figure을 Plot하는 code
% 

clc, clear
format long

filename = 'new_soldier.xlsx'; % 엑셀 파일의 위치를 Access 
xl = xlsread(filename);
%%
singleres = xl(1:3, 1:6);
awmr_pool = xl(12:14, 1:15);
awmr_mean = xl(20:22, 1:15);
awmr_weighted = xl(27:29, 1:15);
octree_pool = xl(35:37, 1:15);
octree_mean = xl(43:45, 1:15);
octree_weighted = xl(50:52,1:15);
%%
figure(1)
plot(singleres(1,:), singleres(2,:),'-o', LineWidth = 2)
hold on;
plot(awmr_pool(1,:), awmr_pool(2,:),'-o', LineWidth = 2);
plot(awmr_mean(1,:), awmr_mean(2,:),'-o', LineWidth = 2);
plot(awmr_weighted(1,:), awmr_weighted(2,:),'-o', LineWidth = 2);
plot(octree_pool(1,:), octree_pool(2,:),'-o', LineWidth = 2);
plot(octree_mean(1,:), octree_mean(2,:),'-o', LineWidth = 2);
plot(octree_weighted(1,:), octree_weighted(2,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','AWMR P', 'AWMR M', 'AWMR W',...
    'OCT P', 'OCT M','OCT W')
%%
figure(2)
plot(singleres(1,:), singleres(3,:),'-o', LineWidth = 2)
hold on;
plot(awmr_pool(1,:), awmr_pool(3,:),'-o', LineWidth = 2);
plot(awmr_mean(1,:), awmr_mean(3,:),'-o', LineWidth = 2);
plot(awmr_weighted(1,:), awmr_weighted(3,:),'-o', LineWidth = 2);
% plot(octree_pool(1,:), octree_pool(3,:),'-o', LineWidth = 2);
% plot(octree_mean(1,:), octree_mean(3,:),'-o', LineWidth = 2);
% plot(octree_weighted(1,:), octree_weighted(3,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res',...
    'AWMR P', 'AWMR M', 'AWMR W',...
    'OCT P', 'OCT M','OCT W')
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
title('dataset: soldier, GT: original GT mesh')
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
title('dataset: soldier, GT: voxelsize 1.5 singleres')