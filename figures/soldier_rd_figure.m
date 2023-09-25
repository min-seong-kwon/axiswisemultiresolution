% @ SPL 석사과정생 권민성 
% 엑셀파일로부터 Figure을 Plot하는 code
% 

clc, clear
format long

filename = 'STANFORD_0924.xlsx'; % 엑셀 파일의 위치를 Access 
xl = xlsread(filename,4);
%%
singleres = xl(1:3,1:3);
awmr = xl(7:9, 1:end);
octree = xl(13:15, 1:end);
% awmr_v2 = xl(19:21, 1:8);
% awmr_v2_true = xl(25:27, 1:8);
% octree_v2 = xl(31:33, 1:8);
%%
figure(3)
plot(singleres(1,:), singleres(2,:),'-o', LineWidth = 2)
hold on;
plot(awmr(1,:), awmr(2,:),'-o', LineWidth = 2);
plot(octree(1,:), octree(2,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','AWMR', 'Octree')
title('dataset: soldier, target: 원본 메시')
%%
figure(4)
plot(singleres(1,:), singleres(3,:),'-o', LineWidth = 2)
hold on;
plot(awmr(1,:), awmr(3,:),'-o', LineWidth = 2);
plot(octree(1,:), octree(3,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','AWMR', 'Octree')
title('dataset: armadillo, target: voxelsize 2.0 singleres')

%%
figure(5)
plot(singleres(1,:), singleres(3,:),'-o', LineWidth = 2)
hold on;
plot(awmr(1,:), awmr(3,:),'-o', LineWidth = 2);
plot(awmr_v2(1,:), awmr_v2(3,:),'-o', LineWidth = 2);
plot(awmr_v2_true(1,:), awmr_v2_true(3,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','AWMR', 'AWMR_v2', 'AWMR_v2_true')
title('dataset: dragon, target: voxelsize 0.001 singleres')
%%
figure(6)
plot(singleres(1,:), singleres(3,:),'-o', LineWidth = 2)
hold on;
plot(octree(1,:), octree(3,:),'-o', LineWidth = 2);
plot(octree_v2(1,:), octree_v2(3,:),'-o', LineWidth = 2);
hold off
xlabel('Mesh File Size (KB)')
ylabel('Mean Chamfer Distance')
legend('Single Res','Octree', 'Octree_v2', 'AWMR_v2_true')
title('dataset: dragon, target: voxelsize 0.001 singleres')