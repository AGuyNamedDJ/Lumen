require("dotenv").config();
const express = require("express");
const morgan = require("morgan");
const axios = require('axios'); // Import axios for making requests to the Flask server
const jwt = require('jsonwebtoken');
const { getUserById } = require('../db/helperFunctions/user');
const JWT_SECRET = process.env.JWT_SECRET;

const app = express();
app.use(express.json()); // Ensure you can parse JSON bodies
app.use(morgan('combined')); // Use morgan for logging

const apiRouter = express.Router();

// JWT Middleware for authentication
apiRouter.use(async (req, res, next) => {
    const prefix = 'Bearer ';
    const auth = req.header('Authorization');

    if (!auth) {
        console.log('Authorization header is missing');
        next();
    } else if (auth.startsWith(prefix)) {
        const token = auth.slice(prefix.length);
        try {
            console.log('Received token:', token);
            const parsedToken = await jwt.verify(token, JWT_SECRET);
            console.log('Parsed token:', parsedToken);
            const id = parsedToken && parsedToken.id;
            console.log('Extracted ID from token:', id);
            if (id) {
                req.user = await getUserById(id);
                console.log('Fetched user:', req.user);
                next();
            } else {
                console.error('ID missing in parsed token');
                res.status(400).send({ error: 'Invalid token' });
            }
        } catch (error) {
            console.error('Token verification error:', error);
            res.status(401).send({ error: 'Invalid token' });
        }
    } else {
        next({
            name: 'AuthorizationHeaderError',
            message: `Authorization token must start with ${prefix}`
        });
    }
});

// Handle GET request to '/api' to check if API is running
apiRouter.get("/", (req, res) => {
    res.json({ message: 'API is running' });
});

// Route to handle OpenAI requests via Flask backend
apiRouter.post('/openai', async (req, res) => {
    const { message } = req.body;
    console.log("POST /openai - Request received with message:", message);

    try {
        const response = await axios.post('https://lumen-back-end-flask.onrender.com/conversation', { message });
        console.log("POST /openai - OpenAI response:", response.data);
        res.status(200).json(response.data);
    } catch (error) {
        console.error("POST /openai - Error communicating with OpenAI:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// General error handling middleware
apiRouter.use((error, req, res, next) => {
    res.status(500).send({ message: error.message });
});

// Importing and using routers
const conversationsRouter = require('./helperFunctions/conversations');
const finnhubRoutes = require('./finnhubAPI');
const loginRouter = require('./helperFunctions/login');
const lumen1Router = require('./lumen_1');
const messagesRouter = require('./helperFunctions/messages');
const signupRouter = require('./helperFunctions/signup');
const userRouter = require('./helperFunctions/user');
apiRouter.use('/conversations', conversationsRouter);
apiRouter.use('/finnhub', finnhubRoutes);
apiRouter.use('/login', loginRouter);
apiRouter.use('/lumen_1', lumen1Router);
apiRouter.use('/messages', messagesRouter);
apiRouter.use('/signup', signupRouter);
apiRouter.use('/user', userRouter);

module.exports = { apiRouter };