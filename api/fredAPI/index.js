const express = require('express');
const fredAPIRouter = express.Router();

const averageHourlyEarningsRouter = require('./averageHourlyEarningsData');
const consumerSentimentRouter = require('./consumerSentimentData');
const coreInflationRouter = require('./coreInflationData');
const cpiRouter = require('./fredCPIData');
const gdpRouter = require('./gdpData');
const interestRateRouter = require('./interestRateData');
const laborForceParticipationRouter = require('./laborForceParticipationRateData');
const nonfarmPayrollEmploymentRouter = require('./nonfarmPayrollEmploymentData');
const pceRouter = require('./pceData');
const ppiRouter = require('./ppiData');
const unemploymentRateRouter = require('./unemploymentRateData');
fredAPIRouter.use('/average-hourly-earnings', averageHourlyEarningsRouter);
fredAPIRouter.use('/consumer-sentiment', consumerSentimentRouter);
fredAPIRouter.use('/core-inflation', coreInflationRouter);
fredAPIRouter.use('/cpi', cpiRouter);
fredAPIRouter.use('/gdp', gdpRouter);
fredAPIRouter.use('/interest-rate', interestRateRouter);
fredAPIRouter.use('/labor-force-participation', laborForceParticipationRouter);
fredAPIRouter.use('/nonfarm-payroll-employment', nonfarmPayrollEmploymentRouter);
fredAPIRouter.use('/pce', pceRouter);
fredAPIRouter.use('/ppi', ppiRouter);
fredAPIRouter.use('/unemployment-rate', unemploymentRateRouter);

module.exports = fredAPIRouter;
