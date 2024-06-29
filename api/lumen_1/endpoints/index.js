const express = require('express');
const endpointsRouter = express.Router();

// Importing specific endpoint routers
const predictRouter = require('./predict');

// Using specific endpoint routers
endpointsRouter.use('/predict', predictRouter);

module.exports = endpointsRouter;