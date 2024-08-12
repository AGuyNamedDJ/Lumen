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
const importAllSPXCSVFiles = require('./db/fetchS3DataForSPX');
const importAllSPYCSVFiles = require('./db/fetchS3DataForSPY');
const importAllVIXCSVFiles = require('./db/fetchS3DataForVIX');

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
};

// Function to import CSV data
async function importCSVData() {
    try {
        await importAllSPXCSVFiles();
        await importAllSPYCSVFiles();
        await importAllVIXCSVFiles();
        logger.info('CSV import completed');
    } catch (error) {
        logger.error('Error importing CSV data.', error);
    }
};

// Sequentially run CSV import and then WebSocket
async function startServer() {
    await importCSVData(); 
    await startWebSocket();
    logger.info('Server initialization completed.');
};

// Import Historic fredAPI Data
const { fetchAllHistoricFredAPIData } = require('./dataFetching/fredAPI/historic');
fetchAllHistoricFredAPIData();

// Import Live fredAPI Data
const {scheduleAverageHourlyEarningsUpdates} = require('./dataFetching/fredAPI/live/liveAverageHourlyEarningsData');
const {scheduleConsumerConfidenceUpdates} = require('./dataFetching/fredAPI/live/liveConsumerConfidenceData');
const {scheduleConsumerSentimentUpdates} = require('./dataFetching/fredAPI/live/liveConsumerSentimentData');
const {scheduleCoreInflationUpdates} = require('./dataFetching/fredAPI/live/liveCoreInflationData');
const {scheduleCPIUpdates} = require('./dataFetching/fredAPI/live/liveCPIData');
const {scheduleGDPUpdates} = require('./dataFetching/fredAPI/live/liveGDPData');
const {scheduleIndustrialProductionUpdates} = require('./dataFetching/fredAPI/live/liveIndustrialProductionData');
const {scheduleInterestRateUpdates} = require ('./dataFetching/fredAPI/live/liveInterestRateData');
const {scheduleLaborForceParticipationRateUpdates} = require('./dataFetching/fredAPI/live/liveLaborForceParticipationRateData');
const {scheduleNonfarmPayrollEmploymentUpdates} = require('./dataFetching/fredAPI/live/liveNonFarmPayrollEmploymentData');
const {schedulePCEUpdates} = require('./dataFetching/fredAPI/live/livePCEData');
const {schedulePPIUpdates} = require('./dataFetching/fredAPI/live/livePPIData');
const {scheduleUnemploymentRateUpdates} = require('./dataFetching/fredAPI/live/liveUnemploymentRateData');

async function importLiveFredAPIData() {
    try {
        scheduleAverageHourlyEarningsUpdates();
        scheduleConsumerConfidenceUpdates();
        scheduleConsumerSentimentUpdates();
        scheduleCoreInflationUpdates();
        scheduleCPIUpdates();
        scheduleGDPUpdates();
        scheduleIndustrialProductionUpdates();
        scheduleInterestRateUpdates();
        scheduleLaborForceParticipationRateUpdates();
        scheduleNonfarmPayrollEmploymentUpdates();
        schedulePCEUpdates();
        schedulePPIUpdates();
        scheduleUnemploymentRateUpdates();

        logger.info('fredAPI Data import completed.');
    } catch (error) {
        logger.error ('Error importing fredAPI data!');
    }
};

importLiveFredAPIData();

// Import Live VIX Data
const { scheduleVIXUpdates } = require('./dataFetching/yahooFinance/scheduleVIXData');
scheduleVIXUpdates();

// Import Indicators
const {scheduleAggregatesDataUpdates} = require('./dataFetching/indicators/schedule/aggregates');
const {scheduleEMAUpdates} = require('./dataFetching/indicators/schedule/emaIndicators');
const {scheduleMACDUpdates} = require('./dataFetching/indicators/schedule/macdIndicators');
const {scheduleRSIUpdates} = require('./dataFetching/indicators/schedule/rsiIndicators');
const {scheduleSMAUpdates} = require('./dataFetching/indicators/schedule/smaIndicators');
async function importIndicators() {
    try {
        scheduleAggregatesDataUpdates();
        scheduleEMAUpdates();
        scheduleMACDUpdates();
        scheduleRSIUpdates();
        scheduleSMAUpdates();
        logger.info('Indicator imports completed.');
    } catch (error) {
        logger.error ('Error importing fredAPI data!');
    }
};

importIndicators();

// Start WebSocket on Server Start
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