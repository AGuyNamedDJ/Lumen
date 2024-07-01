const express = require('express');
const endpointsRouter = express.Router();

// Endpoint routers
const dataRouter = require('./data');
const evaluateRouter = require('./evaluate');
const predictRouter = require('./predict');
const trainRouter = require('./train');

endpointsRouter.use('/data', dataRouter);
endpointsRouter.use('/evaluate', evaluateRouter);
endpointsRouter.use('/predict', predictRouter);
endpointsRouter.use('./train', trainRouter);

// Export
module.exports = endpointsRouter;