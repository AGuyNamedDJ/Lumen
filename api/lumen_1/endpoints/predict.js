const express = require('express');
const predictRouter = express.Router();
const axios = require('axios');

// Use the Render.com URL for your Python service
const PREDICTION_SERVICE_URL = 'https://lumen-0q0f.onrender.com/predict';

// Middleware for logging requests to /predict
predictRouter.use((req, res, next) => {
    next();
});

// Endpoint to handle prediction requests
predictRouter.post('/', async (req, res, next) => {
    try {
        const { input_data } = req.body;
        if (!input_data) {
            return res.status(400).json({ success: false, message: 'Input data is required' });
        }

        // Make a request to the Python service for prediction
        const response = await axios.post(PREDICTION_SERVICE_URL, { input_data });
        const prediction = response.data;

        res.json({ success: true, prediction });
    } catch (error) {
        next(error);
    }
});

module.exports = predictRouter;