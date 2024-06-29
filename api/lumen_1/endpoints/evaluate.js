const express = require('express');
const evaluateRouter = express.Router();
const axios = require('axios');

// Use the Render.com URL for your Python service
const EVALUATION_SERVICE_URL = 'https://lumen-0q0f.onrender.com/evaluate';

// Middleware for logging requests to /evaluate
evaluateRouter.use((req, res, next) => {
    console.log(`Received request at /evaluate: ${new Date().toISOString()}`);
    next();
});

// Endpoint to handle evaluation requests
evaluateRouter.post('/', async (req, res, next) => {
    try {
        const { evaluation_data } = req.body;
        if (!evaluation_data) {
            return res.status(400).json({ success: false, message: 'Evaluation data is required' });
        }

        // Make a request to the Python service for evaluation
        const response = await axios.post(EVALUATION_SERVICE_URL, { evaluation_data });
        const result = response.data;

        res.json({ success: true, result });
    } catch (error) {
        next(error);
    }
});

// Export
module.exports = evaluateRouter;