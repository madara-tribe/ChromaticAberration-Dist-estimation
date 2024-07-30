function [kr kg kb] = bkernels(depth)

% Alpha
frat = 26/44;

k1 = fspecial('disk',depth);
k2a = fspecial('disk',depth*frat);

k2 = zeros(size(k1));
k2(depth+1,depth+1) = 1;
k2 = conv2(k2,k2a,'same');

% Normalize wrt standard aperture
k1 = k1 / sum(k1(:)); 
k2 = k2 / max(k2(:))*max(k1(:));

kr = k1; kg = k2; kb = k1;
