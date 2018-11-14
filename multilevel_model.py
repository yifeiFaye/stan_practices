## PyStan model
## Multilevel model

"""
background: Radon is a radioactive gas that enters homes through contact points with the 
ground, and it's the primary cause of lung cancer in non-smokers. 

The EPA did a study of radon levels in 80,000 houses. Two important predictors:

measurement in basement or first floor (radon higher in basements)
county uranium level (positive correlation with radon levels)
We will focus on modeling radon levels in Minnesota.

The hierarchy in this example is households within county.

"""

# import data from file and extract Minnesote's data
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os

import pystan

os.getcwd()
srrs2 = pd.read_csv("srrs2.csv")
# strip off the white space in column names
srrs2.columns = srrs2.columns.map(str.strip)
# assign() creates a new column
srrs_mn = srrs2.assign(fips = srrs2.stfips * 1000 + srrs2.cntyfips)[srrs2.state == "MN"]

# read county level predictor uranium, by combining two variables 
cty = pd.read_csv("cty.csv")
cty_mn = cty[cty.st == "MN"].copy()
cty_mn['fips'] = cty_mn['stfips'] * 1000 + cty_mn['ctfips']

# merge the data and dedupe
srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on = 'fips')
pd.value_counts(srrs_mn.idnum).describe() # there's duplicate
srrs_mn = srrs_mn.drop_duplicates(subset = 'idnum')
u = np.log(srrs_mn.Uppm)
n = len(srrs_mn)

# create a lookup table for each unique county, for indexing
srrs_mn.county = srrs_mn.county.str.strip()
mn_county = srrs_mn.county.unique()
counties = len(mn_county)

# create county_code in srrs_mn using dict
county_lookup = dict(zip(mn_county, range(len(mn_county))))
county = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values 
srrs_mn.head()

# specify radon
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values

floor_measure = srrs_mn.floor.values

plt.hist(log_radon, bins = 25)
plt.show()

"""
conventional approach:
complete pooling : treat all counties the same, and estimate a single radon level
no pooling: model radon in each county independently
"""
# complete pooling model
# y[i] = beta[1] + beta[2] * x[i] + sigma
pooled_data = """
data {
	int<lower = 0> N;
	vector[N] x;
	vector[N] y;
}
"""
# suspect this sigma is universal
pooled_parameter = """
parameters {
	vector[2] beta;
	real<lower = 0> sigma;
}
"""

pooled_model = """
model {
	y ~ normal(beta[1] + beta[2] * x, sigma);
}
"""

# pass the code, data, parameters to stan function. 
# specify the number of iteration and number of chains

pooled_data_dict = {'N': len(log_radon),
					'x': floor_measure,
					'y': log_radon}

pooled_fit = pystan.stan(model_code = pooled_data + pooled_parameter + pooled_model,
						data = pooled_data_dict,
						iter = 1000,
						chains = 2)

pooled_sample = pooled_fit.extract(permuted=True)
b0, m0 = pooled_sample['beta'].T.mean(1)

plt.scatter(srrs_mn.floor, np.log(srrs_mn.activity + 0.1))
xvals = np.linspace(-0.2, 1.2)
plt.plot(xvals, m0 * xvals + b0, 'r--')
plt.show()

# diagnosis:
print(pooled_fit.stansummary())
"""
Inference for Stan model: anon_model_e2d617eefdef8e123087bf43df777a15.
2 chains, each with iter=1000; warmup=500; thin=1; 
post-warmup draws per chain=500, total post-warmup draws=1000.

          mean se_mean     sd   2.5%    25%    50%    75%  97.5%  n_eff   Rhat
beta[1]   1.36  9.4e-4   0.03   1.31   1.34   1.36   1.38   1.42    846    1.0
beta[2]  -0.59  2.4e-3   0.07  -0.72  -0.63  -0.59  -0.54  -0.45    826    1.0
sigma     0.79  6.5e-4   0.02   0.76   0.78   0.79    0.8   0.83    819    1.0
lp__    -243.7    0.06   1.22 -246.7 -244.3 -243.4 -242.8 -242.3    391    1.0

Samples were drawn using NUTS at Thu Nov  8 15:48:47 2018.
For each parameter, n_eff is a crude measure of effective sample size,
and Rhat is the potential scale reduction factor on split chains (at 
convergence, Rhat=1).
"""
pooled_fit.plot()
plt.show()

# complete no-pooling model
# fit a separate model for each county
# the county level independent variable is the mean for each county (the intercept)
# the shared parameter across county is just the standard deviation of error and 
# beta/the coefficient of floor_measure

unpooled_model = """data {
	int<lower = 0> N;
	int<lower = 0, upper = 85> county[N];
	vector[N] x;
	vector[N] y;
}

parameters {
	vector[85] a;
	real beta;
	real<lower = 0, upper = 100> sigma;
}

transformed parameters{
	vector[N] yhat;

	for ( i in 1:N)
		yhat[i] <- beta * x[i] + a[county[i]];
}

model {
	y ~ normal(yhat, sigma);
}
"""

unpooled_data_dict = {'N': len(log_radon),
					'x': floor_measure,
					'y': log_radon,
					'county': county + 1}

# county need to add one because stan count starting from 1
unpooled_fit = pystan.stan(model_code = unpooled_model,
						data = unpooled_data_dict,
						iter = 1000,
						chains = 2)

unpooled_fit.plot() # way too many coefficients, dirrect plotting is not helpful at all. 
plt.show()

a = unpooled_fit['a'] # a contains 1000 iteration of county level a with length of 85
beta = unpooled_fit['beta'] # beta contains 1000 iteration of beta estimation, just 1000 by 1 array

unpooled_estimates = pd.Series(np.mean(unpooled_fit['a'], axis = 0), index = mn_county)
unpooled_se = pd.Series(np.std(unpooled_fit['a'], axis = 0), index = mn_county)

order = unpooled_estimates.sort_values().index
plt.scatter(range(len(unpooled_estimates)), unpooled_estimates[order])
for i, m, se in zip(range(len(unpooled_estimates)), unpooled_estimates[order], unpooled_se[order]):
	plt.plot([i, i], [m - se, m + se], 'b-')

plt.xlim(-1, 86)
plt.ylim(-1, 4)
plt.ylabel('Radon Estimation')
plt.xlabel('Ordered County')
plt.show()

# do some visualization for subset of counties
sample_county = ('LAC QUI PARLE', 'AITKIN', 'KOOCHICHING', 'DOUGLAS', 'CLAY', 'STEARNS', 'RAMSEY', 'ST LOUIS')

fig, axes = plt.subplots(2, 4, figsize = (12, 6), sharey = True, sharex = True)
axes = axes.ravel()
m = np.mean(unpooled_fit['beta'], axis = 0)

for i, c in enumerate(sample_county):
	y = srrs_mn.log_radon[srrs_mn.county == c]
	x = srrs_mn.floor[srrs_mn.county == c]
	axes[i].scatter(x + np.random.randn(len(x))*0.01, y, alpha = 0.4)
	# No pooling model
	b = unpooled_estimates[c]
	# plot both models and data
	xvals = np.linspace(-0.2, 1.2)
	axes[i].plot(xvals, m*xvals + b)
	axes[i].plot(xvals, m0*xvals + b0, 'r--')
	axes[i].set_xticks([0,1])
	axes[i].set_xticklabels(['basement', 'floor'])
	axes[i].set_ylim(-1, 3)
	axes[i].set_title(c)
	if not i%2:
		axes[i].set_ylabel('log radon level')

plt.show()


########
# Multilevel and Hierarchical models
# partial pooling model
########
# A partial pooling model represents a compromise between the pooled and unpooled extremes, approximately
# a weighted average (based on sample size) of the unpooled county estimates and the pooled estimates. 
partial_pooling = """data{
	int<lower = 0> N;
	int<lower = 1, upper = 85> county[N];
	vector[N] y;
}

parameters{
	vector[85] a;
	real mu_a;
	real<lower = 0, upper = 100> sigma_a;
	real<lower = 0, upper = 100> sigma_y;
}

transformed parameters{
	vector[N] y_hat;
	for (i in 1:N)
		y_hat[i] <- a[county[i]];
}

model{
	mu_a ~ normal(0,1);
	a ~ normal(10 * mu_a, sigma_a);

	y ~ normal(y_hat, sigma_y);	
}"""

partial_pool_data = {'N': len(log_radon),
					'county': county + 1,
					'y': log_radon}

partial_pool_fit = pystan.stan(model_code = partial_pooling,
							 data = partial_pool_data,
							 iter = 1000, 
							 chains = 2)

sample_trace = partial_pool_fit['a']

fig, axes = plt.subplots(1, 2, figsize = (14, 6), sharex = True, sharey = True)
samples, counties = sample_trace.shape
jitter = np.random.normal(scale = 0.1, size = counties)

n_county = srrs_mn.groupby('county')['idnum'].count()
unpooled_means = srrs_mn.groupby('county')['log_radon'].mean()
unpooled_sd = srrs_mn.groupby('county')['log_radon'].std()
unpooled = pd.DataFrame({'n': n_county, 
						'm': unpooled_means,
						'sd': unpooled_sd})

unpooled['se'] = unpooled.sd/np.sqrt(unpooled.n)

axes[0].plot(unpooled.n+jitter, unpooled.m, 'b.')
for j, row in zip(jitter, unpooled.iterrows()):
	name, dat = row
	axes[0].plot([dat.n+j, dat.n+j], [dat.m-dat.se, dat.m+dat.se], 'b-')	

axes[0].set_xscale('log')
axes[0].hlines(sample_trace.mean(), 0.9, 100, linestyles = '--')

samples, counties = sample_trace.shape
means = sample_trace.mean(axis = 0)
sd = sample_trace.std(axis = 0)
axes[1].scatter(n_county.values + jitter, means)
axes[1].set_xscale('log')
axes[1].set_xlim(1, 100)
axes[1].set_ylim(0, 3)
axes[1].hlines(sample_trace.mean(), 0.9, 100, linestyles = '--')
for i, n, m, s in zip(jitter, n_county, means, sd):
	axes[1].plot([n+j]*2, [m-s, m+s], 'b-')

plt.show()

# see the unpooled result is really fuzzy and not accurate.

############
# Varying intercept model
# this model allows intercepts to vary across county, according to
# a random effect
# y[i] = a[ji] + beta*x[i] + eta[i]
# eta[i] ~ N(0, sigma square)
# a[ji] ~ N(mu[a], sigma[a]^2)
############

varying_intercept = """
data{
	int<lower = 0> N;
	int<lower = 0> J;
	int<lower = 1, upper = J> county[N];
	vector[N] x;
	vector[N] y;
}
parameters{
	vector[J] a;
	real mu_a;
	real b;
	real <lower = 0, upper = 100> sigma_a;
	real <lower = 0, upper = 100> sigma_y;
}
transformed parameters{
	vector[N] y_hat;

	for (i in 1:N)
		y_hat[i] <- a[county[i]] + x[i]*b;
}
model{
	sigma_a ~ uniform(0, 100);
	a ~ normal(mu_a, sigma_a);
	b ~ normal(0,1);
	sigma_y ~ uniform(0, 100);
	y ~ normal(y_hat, sigma_y);	
}"""

varying_intercept_data = {'N': len(log_radon),
						'J': len(n_county),
						'county': county + 1,
						'x': floor_measure,
						'y': log_radon}

varying_intercept_fit = pystan.stan(model_code = varying_intercept,
									data = varying_intercept_data,
									iter = 1000,
									chains = 2)

a_sample = pd.DataFrame(varying_intercept_fit['a'])
sns.set(style = 'ticks', palette = 'muted', color_codes = True)

# plot the orbital period with horizontal boxes
plt.figure(figsize = (16, 6))
sns.boxplot(data = a_sample, whis = np.inf, color = 'c')
plt.show()

b_sample = pd.DataFrame(varying_intercept_fit['b'])
b_sample.plot()
plt.show()

varying_intercept_fit.plot(pars = ['sigma_a', 'b'])
plt.show()

np.exp(varying_intercept_fit['b'].mean())
# this could be understand as houses with basement has roughly half
# of radon level compared with houses without basement

xvals = np.arange(2)
bp = varying_intercept_fit['a'].mean(axis = 0)
mp = varying_intercept_fit['b'].mean()
for bi in bp:
	plt.plot(xvals, mp * xvals + bi, 'bo-', alpha = 0.4)

plt.xlim(-0.1, 1.1)
plt.show()
# this shows that varying intercept provides more reasonable estimates
# than either the pooled or unpooled model, at least for counties with 
# small sample size

# let's plot some counties
fig, axes = plt.subplots(2, 4, figsize = (12, 6), sharey = True, sharex = True)
axes = axes.ravel()

for i, c in enumerate(sample_county):

	# plot county data
	y = srrs_mn.log_radon[srrs_mn.county == c]
	x = srrs_mn.floor[srrs_mn.county == c]
	axes[i].scatter(x + np.random.randn(len(x))*0.01, y, alpha = 0.4)

	# plot No pooling model
	m, b = unpooled_estimates[['floor', c]]

	xvals = np.linspace(-0.2, 1.2)
	# unpooled estimate
	axes[i].plot(xvals, m*xvals + b)

	# pooled estimate
	axes[i].plot(xvals, m0*xvals + b0, 'r--')

	# partial pooling estimate
	axes[i].plot(xvals, mp*xvals + bp[county_lookup[c]], 'k:')

	axes[i].set_xticks([0,1])
	axes[i].set_xticklabels(['basement', 'floor'])
	axes[i].set_ylim(-1, 3)
	axes[i].set_title(c)

	if not i % 2:	
		axes[i].set_ylabel('log_radon level')


plt.show()

#############
# Varying slope model
# y[i] = a + beta[j, i]*x[i] + e[i]
#