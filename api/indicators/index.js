const express = require('express');
const indicatorsRouter = express.Router();

const emaRouter = require('./emaIndicators');
const macd_indicators = require('./macdIndicators');
const rsiRouter = require('./rsiIndicators');
const smaRouter = require('./smaIndicators');
indicatorsRouter.use('/ema', emaRouter);
indicatorsRouter.use('/macd', macd_indicators);
indicatorsRouter.use('/rsi', rsiRouter);
indicatorsRouter.use('/sma', smaRouter);

module.exports = indicatorsRouter;