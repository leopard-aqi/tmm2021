%%%%%%%%%%%%%%%%ЪЇец%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function im22=img2refocusdis(LF,Slope)     
addpath('refocustool');
len=length(Slope);
for i=1:len
[ShiftImg] = LFFiltShiftSum( LF, Slope(i) );
ShiftImg=ShiftImg(:,:,1:3);
ShiftImg2(:,:,:,i)=ShiftImg;
end                                  
im22(:,:,:,1)=ShiftImg2(:,:,:,1);               
im22(:,:,:,2)=ShiftImg2(:,:,:,2); 
