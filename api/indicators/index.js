const express = require('express');
const indicatorsRouter = express.Router();

const emaRouter = require('./emaIndicators');
const smaRouter = require('./smaIndicators');
indicatorsRouter.use('/ema', emaRouter);
indicatorsRouter.use('/sma', smaRouter);

module.exports = indicatorsRouter;