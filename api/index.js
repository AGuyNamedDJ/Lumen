require("dotenv").config();
const express = require("express");
const morgan = require("morgan");
const axios = require('axios');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const { getUserById } = require('../db/helperFunctions/user');
const { getSPXPrice } = require('./finnhubAPI/finnhubAPI');
const { getCurrentDateTime } = require('../utils/getCurrentDataTime');

const JWT_SECRET = process.env.JWT_SECRET;

const app = express();
app.use(express.json());
app.use(morgan('combined'));

const apiRouter = express.Router();

apiRouter.use(cors({
    origin: ['http://localhost:3000', 'https://lumen-0q0f.onrender.com', 'https://lumen-back-end-flask.onrender.com'],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    credentials: true,
    allowedHeaders: ['Content-Type', 'Authorization'],
    preflightContinue: false,
    optionsSuccessStatus: 204
}));

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
        const currentDateTime = getCurrentDateTime();
        const enhancedMessage = `Current Date and Time: ${currentDateTime}\nUser Query: ${message}`;

        const response = await axios.post('https://lumen-back-end-flask.onrender.com/conversation', { message: enhancedMessage });
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

// Endpoint to get current SPX price
apiRouter.get('/spx-price', async (req, res) => {
    try {
        const price = await getSPXPrice();
        res.json({ price });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch SPX price' });
    }
});

// Importing and using routers
const conversationsRouter = require('./helperFunctions/conversations');
const fredApiRouter = require('./fredAPI/index');
const finnhubRouter = require('./finnhubAPI');
const indicatorsRouter = require('./indicators/index');
const loginRouter = require('./helperFunctions/login');
const lumen1Router = require('./lumen_1');
const messagesRouter = require('./helperFunctions/messages');
const polygonRouter = require('./polygonApi/index');
const scrapersRouter = require('./scrappers/index');
const signupRouter = require('./helperFunctions/signup');
const spxPriceRouter = require('./finnhubAPI/spxPriceRouter');
const userRouter = require('./helperFunctions/user');
const yahooFinanceRouter = require('./yahooFinance/index');
apiRouter.use('/conversations', conversationsRouter);
apiRouter.use('/fred', fredApiRouter);
apiRouter.use('/finnhub', finnhubRouter);
apiRouter.use('/indicators', indicatorsRouter);
apiRouter.use('/login', loginRouter);
apiRouter.use('/lumen_1', lumen1Router);
apiRouter.use('/messages', messagesRouter);
apiRouter.use('/polygon', polygonRouter);
apiRouter.use('/scrapers', scrapersRouter);
apiRouter.use('/signup', signupRouter);
apiRouter.use('/spx-price', spxPriceRouter);
apiRouter.use('/user', userRouter);
apiRouter.use ('/yahoo-finance', yahooFinanceRouter);

module.exports = { apiRouter };