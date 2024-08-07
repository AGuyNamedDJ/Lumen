const express = require('express');
const fredAPIRouter = express.Router();

const cpiRouter = require('./fredCPIData');
const ppiRouter = require('./ppiData');
fredAPIRouter.use('/cpi', cpiRouter);
fredAPIRouter.use('/ppi', ppiRouter);

module.exports = fredAPIRouter;