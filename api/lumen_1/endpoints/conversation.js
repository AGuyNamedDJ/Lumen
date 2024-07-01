const express = require('express');
const conversationRouter = express.Router();
const axios = require('axios');
require('dotenv').config();

const AI_BACKEND_URL = process.env.AI_BACKEND_URL || 'http://localhost:5000';  // URL to your Python backend

// Middleware
conversationRouter.use((req, res, next) => {
    console.log(`Received request at /conversation: ${new Date().toISOString()}`);
    next();
});

// Endpoint to handle conversation requests
conversationRouter.post('/', async (req, res, next) => {
    try {
        const { message } = req.body;
        if (!message) {
            return res.status(400).json({ success: false, message: 'User message is required' });
        }

        // Make a request to Lumen
        const response = await axios.post(`${AI_BACKEND_URL}/conversation`, { message });
        const aiResponse = response.data.response;

        res.json({ success: true, response: aiResponse });
    } catch (error) {
        console.error('Error handling conversation:', error);
        next(error);
    }
});

module.exports = conversationRouter;