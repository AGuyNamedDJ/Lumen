const express = require('express');
const fredAPIRouter = express.Router();

const cpiRouter = require('./fredCPIData');
fredAPIRouter.use('/cpi', cpiRouter);

module.exports = fredAPIRouter;