const express = require('express');
const lumen2Router = express.Router();

// Importing endpoint routers
const endpointsRouter = require('./endpoints');

// Using endpoint routers
lumen2Router.use('/endpoints', endpointsRouter);

module.exports = lumen2Router;