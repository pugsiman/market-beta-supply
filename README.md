## Description
Small predictive model built over an implementation of Welch's [Simply Better Market Betas](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3371240).

A populator script creates datasets for all current estimator values in NASDAQ listings (approximating for the overall supply in the market), which is then used by the model to track the time varying supply of high beta.

Short periods following changes in supply lower tend to lead to more volatile returns in highly risky and liquid stocks ("junk stocks"), which can be helpful in optimization of hedging, utilized in a form of dispersion trading, and more.

The model also includes a measure called beta dispersion, a measure of market vulnarablity that can be used for market timing, as outlined by Kuntz. Interestingly, most of the variation in Kuntz' measure is by the same segment of beta I'm tracking. That would challenge the theoretical underpinning of the measure: Does supply go higher as holding stocks become riskier, or does demand for risk creates crowding in positioning, in turn making the market riskier?

## Setup
To run you will first need to clone the repo. Then, create your datasets:

`poetry run utils/beta_dists_creator.py`

You can create a cronjob automate the creation of these daily.

From here the main script can be run to create visualization of the data.

## Example
![image](https://github.com/pugsiman/market-beta-supply/assets/12158433/1aa1a3ad-cee1-4baf-a6ac-bd856305aeb8)

## References
* Welch, Ivo, Simply Better Market Betas (june 13, 2021). Available at SSRN: https://ssrn.com/abstract=3371240 or http://dx.doi.org/10.2139/ssrn.3371240

* Kuntz, Laura-Chloé, Beta Dispersion and Market Timing (2020). Deutsche Bundesbank Discussion Paper No. 46/2020, Available at SSRN: https://ssrn.com/abstract=3684268 or http://dx.doi.org/10.2139/ssrn.3684268
