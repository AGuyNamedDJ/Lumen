require('dotenv').config();
const express = require('express');
const morgan = require('morgan');
const cors = require('cors');
const cron = require('node-cron');
const axios = require('axios');
const app = express();
const winston = require('winston');

// Setup Winston logger
const logger = winston.createLogger({
    level: 'debug',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ timestamp, level, message }) => {
            return `${timestamp} [${level.toUpperCase()}]: ${message}`;
        })
    ),
    transports: [
        new winston.transports.Console()
    ],
});

// Import project dirs
const { apiRouter } = require('./api/index');
const { client } = require('./db/index');
const { handleWebSocket, restartWebSocket } = require('./api/finnhubAPI/finnhubWebsocket');
const importAllCSVFiles = require('./db/fetchS3Data');

// Middleware
app.use(cors({
    origin: ['http://localhost:3000', 'https://lumen-0q0f.onrender.com', 'https://lumen-back-end-flask.onrender.com'],
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    credentials: true,
    allowedHeaders: ['Content-Type', 'Authorization'],
    preflightContinue: false,
    optionsSuccessStatus: 204
}));

app.use(express.json());
app.use(morgan('combined', {
    stream: {
        write: (message) => logger.info(message.trim())
    }
}));

// Log the JWT_SECRET
logger.debug('JWT_SECRET:', process.env.JWT_SECRET);

// Function to start WebSocket
async function startWebSocket() {
    try {
        handleWebSocket();
        logger.info('WebSocket connection started');
    } catch (error) {
        logger.error('Error starting WebSocket:', error);
    }
}

// Function to import CSV data
async function importCSVData() {
    try {
        await importAllCSVFiles();
        logger.info('CSV import completed');
    } catch (error) {
        logger.error('Error importing CSV data:', error);
    }
}

// Sequentially run CSV import and then WebSocket
async function startServer() {
    await importCSVData(); 
    await startWebSocket();
    logger.info('Server initialization completed');
}

// Import CSV Data and Start WebSocket on Server Start
startServer();

// Schedule WebSocket restart every 15 minutes
cron.schedule('*/15 * * * *', () => {
    logger.info('Cron job triggered: Restarting WebSocket connection...');
    restartWebSocket();
});

// Keep the server alive with periodic requests
const URL = `https://lumen-0q0f.onrender.com`;

cron.schedule('*/5 * * * *', () => {
    logger.info('Sending keep-alive request to the server');
    axios.get(URL)
        .then(response => {
            logger.info('Keep-alive request successful:', response.status);
        })
        .catch(error => {
            logger.error('Keep-alive request failed:', error);
        });
});

// API
app.use('/api', apiRouter);

// Temporary direct route to test login
app.use('/test-login', require('./api/helperFunctions/login'));

// Catch-all route handler
app.get("/", (req, res) => {
    res.send("Server is Running!");
    logger.info('Root path accessed');
});

// Router Handlers
client.connect().then(() => {
    logger.info('Connected to the database');
}).catch(error => {
    logger.error("Unable to connect to database.", error);
    process.exit(1);
});

// Close the database connection when the server stops
process.on('exit', () => {
    logger.info('Closing database connection');
    client.end();
});

// Port
const PORT = process.env.PORT || 3001;
if (process.env.NODE_ENV !== 'test') {
    app.listen(PORT, () => {
        logger.info(`Now running on port ${PORT}`);
    });
}

// Export
module.exports = {
    app,
    client,
};