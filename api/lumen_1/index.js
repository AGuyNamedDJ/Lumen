const express = require('express');
const lumen1Router = express.Router();

// Importing endpoint routers
const endpointsRouter = require('./endpoints');

// Using endpoint routers
lumen1Router.use('/endpoints', endpointsRouter);

module.exports = lumen1Router;