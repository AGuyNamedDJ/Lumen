const express = require('express');
const indicatorsRouter = express.Router();

const emaRouter = require('./emaIndicators');
const rsiRouter = require('./rsiIndicators');
const smaRouter = require('./smaIndicators');
indicatorsRouter.use('/ema', emaRouter);
indicatorsRouter.use('/rsi', rsiRouter);
indicatorsRouter.use('/sma', smaRouter);

module.exports = indicatorsRouter;