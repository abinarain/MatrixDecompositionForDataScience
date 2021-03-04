%% Matrix Decompositions in Data Analysis
% Winter 2019
% Assignment file
% FILL IN THE FOLLOWING INFORMATION
% Name: 
% Student ID:
%

%% Main function
% In Matlab, we can't mix script files and function files, so we'll have 
% a main function called assignment2 that will call the other functions
% that can be added to this file.
%
% For prototyping, it's better to call the functions from Matlab command line, 
% as the functions don't save their working space. 

function assignment() 
  % To read the data, we can use the readtable command
  T = readtable('news.csv');
  A = T.Variables;
  terms = T.Properties.VariableNames;

  %% Task 1
  % FILL IN HERE the code for doing Task 1
  % One round of nmf_als would be like
  [W, H, errs] = nmf(A, 20, 'optFunc', @nmf_als, 'maxiter', 300, ...
                     'repetitions', 1);
  % To plot the errors per iterations, use
  figure, plot(errs);
  title('Convergence of NMF ALS');
  xlabel('Iteration');
  ylabel('||A - WH||_F^2');
  % DO THE OTHER NMF methods similarly and add here code to call them
  % and to do the comparisons. 

  %% Task 2
  % We use B for normalised A
  B = A./sum(sum(A));
  % DO THE NMF

  % If H is an output of some NMF algorihtm, we get the top-10 entries
  % of the first row of H as follows
  h = H(1,:);
  [~, I] = sort(h, 'descend');
  for i=1:10
    fprintf('%s\t%f\n', terms{I(i)}, h(I(i)));
  end

  %% Task 3
  % To compute K-L, we need the z-scores
  Z = normalize(A); % This requires MATLAB 2018a
  [U, S, V] = svds(Z, 20);
  KL = Z*V; % N.B. svds returns correct-sized V
  %% COMPUTE pLSA
  % To compute k-means, use kmeans; Replicates controls the number of 
  % restarts. 
  [idx] = kmeans(KL, 20, 'Replicates', 20);
  % DO K-MEANS for other cases 
  % To calculate the NMI, use
  nmi_KL = nmi_news(idx);
  fprintf('NMI (KL) = %f\n', nmi_KL);
  % DO NMI for other clusterings
end


function [bestW, bestH, bestErrs] = nmf(A, k, varargin) 
% NMF - Boilerplate function for NMF
% 
% [W, H] = NMF(A, k) will compute rank-k NMF of nonnegative A.
% [W, H, errs] = NMF(A, k) will also return a vector of errors per
%   iteration.
% NMF(A, k, ..., 'optFunc', @fn) will use function fn for optimising W
%   and H. Default is nmf_als. 
% NMF(A, k, ..., 'maxiter', m) will do m iterative updates. Default
%   is 300.
% NMF(A, k, ..., 'repetitions', r) will restart the algorithm r
%   times and return the best result. Default is 1.

  p = inputParser;
  addRequired(p, 'A', ...
              @(x) validateattributes(x,{'numeric'}, ...
                                      {'nonnegative', '2d'}));
  addRequired(p, 'k', ...
              @(x) validateattributes(x, {'numeric'}, ...
                                      {'scalar', 'positive', ...
                      'integer'}));
  addParameter(p, 'optFunc', @nmf_als, ...
               @(x) validateattributes(x, {'function_handle'}, {} ...
                                       ));
  addParameter(p, 'maxiter', 300, ...
               @(x) validateattributes(x, {'numeric'}, ...
                                       {'scalar', 'positive', ...
                      'integer'}));
  addParameter(p, 'repetitions', 1, ...
               @(x) validateattributes(x, {'numeric'}, ...
                                       {'scalar', 'positive', ...
                      'integer'}));
  
  parse(p, A, k, varargin{:});
  maxiter = p.Results.maxiter;
  repetitions = p.Results.repetitions;
  optFunc = p.Results.optFunc;
  
  [n, m] = size(A);
  bestErr = Inf;
  
  for r=1:repetitions
      W = rand(n, k);
      H = rand(k, m);
      % Init errs to NaNs
      errs = nan(maxiter, 1);
      for i = 1:maxiter,
          [W, H] = optFunc(A, W, H);
          curr_err = norm(A - W*H, 'fro')^2; % Squared Frobenius
          errs(i) = curr_err;
      end
      if curr_err < bestErr
          bestErr = curr_err;
          bestErrs = errs;
          bestW = W;
          bestH = H;
      end
  end

end

function [W, H] = nmf_als(A, W, H)
% NMF_ALS - ALS updates for NMF

% FILL IN THE UPDATES
    
end

function z = nmi(x, y)
%NMI Computes normalized mutual information metric
% Usage: z = NMI(x, y)
% Input:
%   x, y: two integer vector of the same length 
% Ouput:
%   z: normalized mutual information z= 1 - I(x,y)/H(x,y)
% Written by Pauli Miettinen, based on code by Mo Chen (sth4nth@gmail.com).
% https://de.mathworks.com/matlabcentral/fileexchange/29047-normalized-mutual-information
  assert(numel(x) == numel(y));
  n = numel(x);
  x = reshape(x,1,n);
  y = reshape(y,1,n);
  l = min(min(x),min(y));
  x = x-l+1;
  y = y-l+1;
  k = max(max(x),max(y));
  idx = 1:n;
  Mx = sparse(idx,x,1,n,k,n);
  My = sparse(idx,y,1,n,k,n);
  Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y
  Hxy = -dot(Pxy,log2(Pxy));
  % hacking, to elimative the 0log0 issue
  Px = nonzeros(mean(Mx,1));
  Py = nonzeros(mean(My,1));
  % entropy of Py and Px
  Hx = -dot(Px,log2(Px));
  Hy = -dot(Py,log2(Py));
  % mutual information
  MI = Hx + Hy - Hxy;
  % metric normalized mutual information
  z = 1 - MI/Hxy;
  z = max(0,z);

end

function z = nmi_news(x)
% Computes the NMI between the news ground truth and given clustering
  gt = load('news_ground_truth.txt');
  z = nmi(x, gt);
end 

