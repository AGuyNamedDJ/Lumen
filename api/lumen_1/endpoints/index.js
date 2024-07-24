const express = require('express');
const endpointsRouter = express.Router();

// Endpoint routers
const conversationRouter = require('./conversation');
const dataRouter = require('./data');
const evaluateRouter = require('./evaluate');
const predictRouter = require('./predict');
const trainRouter = require('./train');
endpointsRouter.use('/conversation', conversationRouter);
endpointsRouter.use('/data', dataRouter);
endpointsRouter.use('/evaluate', evaluateRouter);
endpointsRouter.use('/predict', predictRouter);
endpointsRouter.use('./train', trainRouter);

// Export
module.exports = endpointsRouter;