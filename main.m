%load or imread the reference light field image
fprintf('\nstep1:load data.\n');
load(['data/5DLFr']); %lensletr=imread('imr.bmp');im10=permute(reshape(lensletr,[9,ss,9,tt,3]),[1,3,2,4,5]);  
%load or imread the distort light field image
load(['data/5DLFd.mat']);   %lensletd=imread('imd.bmp');im20=permute(reshape(lensletd,[9,ss,9,tt,3]),[1,3,2,4,5]);  

%compute the key referenced refocused-images.
% the parameter of refocused range:the minimun slope, the maxmum slope, and
% the number of the slope step of all refocused images

fprintf('\nstep2:key refocused images extraction.\n');
slopemin=-2; 
slopemax=2;
step_num=20;
[keyslope,keyref]=img2refocusnum(im10,slopemin,slopemax,step_num);

%compute the corresponding distorted refocused-images.
keydis=img2refocusdis(im20,keyslope);   
%predict score 
fprintf('\nstep3:predict score.\n');

spaqua=ComputeSpcgc(squeeze(keyref(:,:,:,1)),squeeze(keydis(:,:,:,1)));
angqua=ComputeSpcgc(squeeze(keyref(:,:,:,2)),squeeze(keydis(:,:,:,2))); 
output_score=spaqua.*0.5+angqua.*0.5

 

