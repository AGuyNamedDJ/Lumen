const express = require('express');
const yahooFinanceRouter = express.Router();

const vixRouter = require('./vix');
yahooFinanceRouter.use('/vix', vixRouter);

module.exports = yahooFinanceRouter;