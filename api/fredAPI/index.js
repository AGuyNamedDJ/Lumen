const express = require('express');
const fredAPIRouter = express.Router();

const averageHourlyEarningsRouter = require('./averageHourlyEarningsData');
const cpiRouter = require('./fredCPIData');
const gdpRouter = require('./gdpData');
const interestRateRouter = require('./interestRateData');
const laborForceParticipationRouter = require('./laborForceParticipationRateData');
const nonfarmPayrollEmploymentRouter = require('./nonfarmPayrollEmploymentData');
const ppiRouter = require('./ppiData');
const unemploymentRateRouter = require('./unemploymentRateData');
fredAPIRouter.use('/average-hourly-earnings', averageHourlyEarningsRouter);
fredAPIRouter.use('/cpi', cpiRouter);
fredAPIRouter.use('/gdp', gdpRouter);
fredAPIRouter.use('/interest-rate', interestRateRouter);
fredAPIRouter.use('/labor-force-participation', laborForceParticipationRouter);
fredAPIRouter.use('/nonfarm-payroll-employment', nonfarmPayrollEmploymentRouter);
fredAPIRouter.use('/ppi', ppiRouter);
fredAPIRouter.use('/unemployment-rate', unemploymentRateRouter);

module.exports = fredAPIRouter;