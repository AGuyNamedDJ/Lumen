const express = require('express');
const indicatorsRouter = express.Router();

const smaRouter = require('./smaIndicators');
indicatorsRouter.use('/sma', smaRouter);

module.exports = indicatorsRouter;