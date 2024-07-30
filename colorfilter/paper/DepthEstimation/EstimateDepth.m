% EstimateDepth: Estimate depth from spectrally-varying defocus.
%
% [dmap lhs] = EstimateDepth(img)
%
%   img: 3-color Observed Image
%   
% Please edit the [kr kg kb] = bkernels(r) function to return 
% the induced blur kernels in each channel, for depth corresponding
% to blur radius r.
% Edit candidateDepths to return a vector of candidate radii r.
%
% Output:
%   dmap: Detected depths. 1 corresponds to no focus, and i+1
%         to depth corresponding to index i in vector returned
%         by candidateDepths.
%   lhs:  Computed log-likelihood/score for each depth at each 
%         location.
function [dmap lhs] = EstimateDepth(img)


%%%%%%%%%%%%%%%%%% Config stuff

NSZ = 15; % Window size
gsm = 1;  % Derivative Scale


% Set the two parameters of alignment
% based on visual inspection (once for
% each set of kernels).
krnls = kalign(10^-3,5);

% Do we need to align blue ? Set to 1
% if the induced blur in blue is diff
% from red.
blueAlign = 0;

%%%%%%%%%%%%%%%%%% Config stuff END

% Construct gradient filters
x = [-round(3*gsm):round(3*gsm)];
gss = 1/sqrt(2*pi)/gsm * exp(- x.^2 / 2 / (gsm^2));
gsx = - (x/gsm^2) .* gss;
gf = gss'*gsx; gf = gf / sqrt(sum(gf(:).^2));
%

numl = size(krnls,1); % Number of Candidate depths

% Create "dummy" kernel
sz = (length(krnls{1,1})-1)/2;
krnls{numl+1,1} = fspecial('disk',sz); krnls{numl+1,2} = krnls{numl+1,1};

% Compute likelihoods!
lhs = lhk(img,krnls,NSZ,gf,blueAlign);

% Normalize by likelihood with dummy kernel
lhs = lhs(:,:,1:numl) ./ repmat(abs(lhs(:,:,end)),[1 1 numl]);

% Estimate depthmap
dmap = zeros(size(img,1),size(img,2));
scr = -Inf*ones(size(img,1),size(img,2));

for i = 1:numl
  lh = lhs(:,:,i);  dmap(lh > scr) = i; scr(lh > scr) = lh(lh > scr);
end;


%================================================================
function lh = lhk(img,krnls,nsz,gf,blueAlign)

flts = {ones(1,nsz); ones(nsz,1); diag(ones(nsz,1)); fliplr(diag(ones(nsz,1)))};
grds = {gf; gf'; (gf+gf')/sqrt(2); (gf+gf')/sqrt(2)};

% Precompute gradients in channel 1
imgder = cell(length(flts),1);
for i = 1:length(grds)
  imgder{i} = conv2(img(:,:,1),grds{i},'same');
end;

% Pad by x
psz = length(krnls{end,1})-1;
im2f = fft2(padimg(img(:,:,2),psz));
if blueAlign == 1
  im3f = fft2(padimg(img(:,:,3),psz));
end;

lh = zeros([size(img,1) size(img,2) length(krnls)]);
for i = 1:length(krnls)

  fprintf('Testing for kernel %d of %d\n',i,length(krnls));
  img2 = img;

  % Convolve in the Fourier domain
  if i > 1
    tmp = real(ifft2(im2f .* psf2otf(krnls{i,1},size(im2f))));
    img2(:,:,2) =  tmp(1:end-psz,1:end-psz);
    if blueAlign == 1
      tmp = real(ifft2(im3f .* psf2otf(krnls{i,2},size(im2f))));
      img2(:,:,3) =  tmp(1:end-psz,1:end-psz);
    end;
  end;
  
  for j = 1:length(flts)
    dr = imgder{j};
    dr(:,:,2) = conv2(img2(:,:,2),grds{j},'same');
    dr(:,:,3) = conv2(img2(:,:,3),grds{j},'same');
    
    % DC Value (useful as starting point for power iterations)
    v = img2;
    for c = 1:3
      v(:,:,c) = conv2(img2(:,:,c),flts{j} / sum(flts{j}(:)),'same');
    end;
    
    % Covariance matrix
    R2 = conv2(dr(:,:,1).^2,flts{j},'same');
    G2 = conv2(dr(:,:,2).^2,flts{j},'same');
    B2 = conv2(dr(:,:,3).^2,flts{j},'same');
  
    RG = conv2(dr(:,:,1).*dr(:,:,2),flts{j},'same');
    RB = conv2(dr(:,:,1).*dr(:,:,3),flts{j},'same');
    GB = conv2(dr(:,:,2).*dr(:,:,3),flts{j},'same');
    lh(:,:,i) = lh(:,:,i) - lnerr(R2,G2,B2,RG,RB,GB,v);
  end;
end;
fprintf('\n');

%================================================================

% Compute Rank 1 approximation error with power iterations
function sc1 = lnerr(R2,G2,B2,RG,RB,GB,v2)

maxiters = 3;
for iters = 1:maxiters
  s = sqrt(sum(v2.^2,3));
  s(s == 0) = 1;
  v = v2 ./ repmat(s,[1 1 3]);
  v2 = v;
  v2(:,:,1) = v(:,:,1).*R2+v(:,:,2).*RG+v(:,:,3).*RB;
  v2(:,:,2) = v(:,:,1).*RG+v(:,:,2).*G2+v(:,:,3).*GB;
  v2(:,:,3) = v(:,:,1).*RB+v(:,:,2).*GB+v(:,:,3).*B2;
end;
    
egv = sum(v2.*v,3);
smv = R2+G2+B2;

sc1 =  smv - egv;


%================================================================

% Pad Image
function imo = padimg(img,psz)

psz = psz/2;

img = [img repmat(img(:,end),[1 psz]) repmat(img(:,1),[1 psz])];
imo = [img; repmat(img(end,:),[psz 1]); repmat(img(1,:),[psz 1])];

%================================================================

% Find k^{r/g} k^{b/g} as described in paper.
function krnls = kalign(mu,fctr)
depths = [0 candidateDepths()];
krnls = cell(length(depths),2);
sz = round(max(depths)*fctr)*2+1;

for i = 1:length(depths)
  if depths(i) > 0
    [kr kg kb] = bkernels(depths(i));
    kr = kr / sum(kr(:)); kg = kg / sum(kg(:)); kb = kb / sum(kb(:));
  else
    kr = 1; kg = 1; kb = 1;
  end;
  krnls{i,1} = getk(kr,kg,sz,mu);
  krnls{i,2} = getk(kr,kb,sz,mu);
end;

function k = getk(k1,k2,sz,mu)

sz = [sz sz];
den1 = abs(psf2otf([-1 1],sz)).^2 / 2 + abs(psf2otf([-1 1]', sz)).^2/2;
k1f = psf2otf(k1,sz);
k2f = psf2otf(k2,sz);

kf = k1f .* conj(k2f) ./(mu * den1 + abs(k2f).^2);
k = fftshift(real(ifft2(kf)));
