const express = require('express');
const fredAPIRouter = express.Router();

const cpiRouter = require('./fredCPIData');
const gdpRouter = require('./gdpData');
const interestRateRouter = require('./interestRateData');
const ppiRouter = require('./ppiData');
const unemploymentRateRouter = require('./unemploymentRateData');
fredAPIRouter.use('/cpi', cpiRouter);
fredAPIRouter.use('/gdp', gdpRouter);
fredAPIRouter.use('/interest-rate', interestRateRouter);
fredAPIRouter.use('/ppi', ppiRouter);
fredAPIRouter.use('/unemployment-rate', unemploymentRateRouter);

module.exports = fredAPIRouter;