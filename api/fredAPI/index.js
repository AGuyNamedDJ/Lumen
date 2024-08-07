const express = require('express');
const fredAPIRouter = express.Router();

const cpiRouter = require('./fredCPIData');
const ppiRouter = require('./ppiData');
const unemploymentRateRouter = require('./unemploymentRateData');
fredAPIRouter.use('/cpi', cpiRouter);
fredAPIRouter.use('/ppi', ppiRouter);
fredAPIRouter.use('/unemploymentrate', unemploymentRateRouter);


module.exports = fredAPIRouter;