const express = require('express');
const trainRouter = express.Router();
const axios = require('axios');

// Use the Render.com URL for your Python service
const TRAINING_SERVICE_URL = 'https://lumen-0q0f.onrender.com/train';

// Middleware for logging requests to /train
trainRouter.use((req, res, next) => {
    console.log(`Received request at /train: ${new Date().toISOString()}`);
    next();
});

// Endpoint to handle training requests
trainRouter.post('/', async (req, res, next) => {
    try {
        const { training_data } = req.body;
        if (!training_data) {
            return res.status(400).json({ success: false, message: 'Training data is required' });
        }

        // Make a request to the Python service for training
        const response = await axios.post(TRAINING_SERVICE_URL, { training_data });
        const result = response.data;

        res.json({ success: true, result });
    } catch (error) {
        next(error);
    }
});

module.exports = trainRouter;