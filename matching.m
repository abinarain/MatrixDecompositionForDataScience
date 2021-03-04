function [matched, X] = matching(idxA,idxB)
%MATCHING Returns a matching of indices
%   matched = MATCHING(idxA, idxB) returns an index array 'matched' which
%   is a permutation of labels of 'idxB' such that the labels in 'matched'
%   match with labels of 'idxA' as well as possible.
%
%   [mached, X] = MATCHING() returns also the assignment matrix X.
    X = bipartite_match(idxA, idxB);
    matched = nan(length(idxB), 1);
    labelsA = unique(idxA);
    labelsB = unique(idxB);
    for i = 1:length(labelsB)
        labelB = labelsB(i);
        labelA = labelsA(X(:,i) == 1);
        matched(idxB == labelB) = labelA;
    end
end

function X = bipartite_match(A, B)
%BIPARTITE_MATCH computes the assignment of B to A
%    X = MATCHING(A, B) returns an assignment matrix X such that label i of
%    A matches to label j of B if and only if X(i,j) = 1.
labelsA = unique(A);
labelsB = unique(B);
m = length(labelsA);
n = length(labelsB);
% Weights, build as a matrix
W = zeros(m,n);
for j=1:n
    for i=1:m
        W(i,j) = -sum(A(B == labelsB(j)) == labelsA(i));
    end
end
% Constraints, build as a dense matrix
% There are n+m constraints and n*m variables
C = zeros(n+m, n*m);
% 1) every column must sum to 1
for i = 1:n
    C(i, (i-1)*m+1:i*m) = 1;
end
% 2) every row must sum to 1
for i = 1:m
    C(n+i, i:m:(n-1)*m+i) = 1;
end
Ceq = ones(n+m, 1);
X = linprog(W(:), [], [], C, Ceq, zeros(n*m, 1), ones(n*m, 1), optimoptions('linprog', 'Display', 'off'));
X = reshape(X, m, n);
end
