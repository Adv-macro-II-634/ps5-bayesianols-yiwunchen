% Problem Set 5
clear
clc
close all
data= readtable('card.csv');
data_new= table2array(data); %change table to array

%Question(1)
lwage = data_new (:,33);
educ  = data_new (:,4);
exper = data_new (:,32);
SMSA  = data_new (:,23);
black  = data_new (:,22);
south  = data_new (:,24);

X = [educ exper SMSA black south];
[n,k] = size(X);
X_cons = [ones(n,1) X];
y=lwage;

ols = fitlm(X,y,'VarNames',{'educ','exper','smsa','black','south','lwage'});
b = ols.Coefficients(:, 1);  
%b_intercept=4.9133; b_educ=0.073807; b_exper=0.039313
%b_smsa=0.16474; b_black=-0.18822; b_south=-0.12905

std_b = ols.Coefficients(:,2); 
%std_b_intercept=0.063121; std_b_educ=0.0035336; std_b_exper=0.0021955
%std_b_smsa=0.015692; std_b_black=0.017768; std_b_south=0.015229 

residual=y-X_cons*table2array(b);
std_residual=std(residual);
sigma=std_residual;
%std_residual=sigma=0.3769

%Question 2 Metropolis-Hastings algorithm (MH)
%part(a) 
% MH
theta = [table2array(b)',sigma]';
var_b = diag(ols.CoefficientCovariance);   

var_theta = [var_b;sigma^2];
var_theta = diag(var_theta);  

r=0.001; %suppose the propotional rate=0.001
var_adj= r*var_theta;

iter=1000; 
theta_update = zeros(iter,length(theta));
theta_adj = zeros(iter,length(theta));
for ii=1:iter   
theta_adj(ii,:)=mvnrnd(theta, var_adj, 1)

[n,k]=size(X_cons)

LogLikeL=(-(n/2)*log(2*pi)-(n/2)*log(theta(k+1))-(1/(2*theta(k+1)))*((y-X_cons*theta(1:k))'*(y-X_cons*theta(1:k))));
LogLikeL_update=(-(n/2)*log(2*pi)-(n/2)*log(theta_adj(ii,k+1)')-(1/(2*theta_adj(ii,k+1)'))*((y-X_cons*theta_adj(ii,1:k)')'*(y-X_cons*theta_adj(ii,1:k)')));

prob_accept = LogLikeL_update -LogLikeL; % acceptance probabilityu 

u = log(rand); % uniform random number
if u <= prob_accept % if accepted
theta_update(ii,:) = theta_adj(ii,:); 
else % if rejected
theta_update(ii,:) = theta'; 
end
end

%Plot histograms of the posterior approximations
histogram(theta_update(:,1))
title('the estimate for the b-intercept');
histogram(theta_update(:,2))
title('the estimate for the b-educ');
histogram(theta_update(:,3))
title('the estimate for the b-exper');
histogram(theta_update(:,4))
title('the estimate for the b-SMSA');
histogram(theta_update(:,5))
title('the estimate for the b-black');
histogram(theta_update(:,6))
title('the estimate for the b-south');
histogram(theta_update(:,7))
title('the estimate for the sigma');


%Part (b)
%Now, the expected value of b_educ=0.06
b=table2array(b)
theta_2 = [b(1,:),0.06,b(3,:),b(4,:),b(5,:),b(6,:),sigma]';

%Now, the variance of b_educ is 4 time the old one
var_b = diag(ols.CoefficientCovariance);   
var_b_2=[var_b(1,:),4*var_b(2,:),var_b(3,:),var_b(4,:),var_b(5,:),var_b(6,:)]'

var_theta_2 = [var_b_2;sigma^2];
var_theta_2 = diag(var_theta_2);  

r=0.001; %suppose the propotional rate=0.001
var_adj_2= r*var_theta_2;

iter=1000; 
theta_update_2 = zeros(iter,length(theta_2));
theta_adj_2 = zeros(iter,length(theta_2));
for ii=1:iter   
theta_adj_2(ii,:)=mvnrnd(theta_2, var_adj_2, 1)

[n,k]=size(X_cons)

LogLikeL=(-(n/2)*log(2*pi)-(n/2)*log(theta_2(k+1))-(1/(2*theta_2(k+1)))*((y-X_cons*theta_2(1:k))'*(y-X_cons*theta_2(1:k))));
LogLikeL_update=(-(n/2)*log(2*pi)-(n/2)*log(theta_adj_2(ii,k+1)')-(1/(2*theta_adj_2(ii,k+1)'))*((y-X_cons*theta_adj_2(ii,1:k)')'*(y-X_cons*theta_adj_2(ii,1:k)')));

prob_accept = LogLikeL_update -LogLikeL; % acceptance probabilityu 

u = log(rand); % uniform random number
if u <= prob_accept % if accepted
theta_update_2(ii,:) = theta_adj_2(ii,:); 
else % if rejected
theta_update_2(ii,:) = theta_2'; 
end
end

 %Plot histograms of the posterior approximations
histogram(theta_update_2(:,1),'FaceColor','g')
title('the estimate for the b-intercept');
histogram(theta_update_2(:,2),'FaceColor','g')
title('the estimate for the b-educ');
histogram(theta_update_2(:,3),'FaceColor','g')
title('the estimate for the b-exper');
histogram(theta_update_2(:,4),'FaceColor','g')
title('the estimate for the b-SMSA');
histogram(theta_update_2(:,5),'FaceColor','g')
title('the estimate for the b-black');
histogram(theta_update_2(:,6),'FaceColor','g')
title('the estimate for the b-south');
histogram(theta_update_2(:,7),'FaceColor','g')
title('the estimate for the sigma');

%Question 3
%Ans: Based on the distribution of the estimates from Q(1) to calculat
%the posterior distribution of the estimates. Thus, the posterior
%distribution is also normal distribution (since the posterior distribution
%propotional to the distribution of the estimates from Q(1)).
