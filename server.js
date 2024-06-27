// Requires
require('dotenv').config();
const express = require('express');
const morgan = require('morgan');
const cors = require('cors');
const app = express();

// Import project dirs
const { client } = require('./db/index');
const handleWebSocket = require('./api/finnhubAPI/finnhubWebsocket');
const importSpecificCSVFile = require('./db/fetchS3Data');

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(morgan('dev'));
app.use(cors());

// Function to start WebSocket
async function startWebSocket() {
    try {
        handleWebSocket();
    } catch (error) {
        console.error('Error starting WebSocket:', error);
    }
}

// Function to import CSV data
async function importCSVData() {
    try {
        await importSpecificCSVFile();
    } catch (error) {
        console.error('Error importing CSV data:', error);
    }
}

// Start server function
async function startServer() {
    // First, import CSV data
    await importCSVData();
    console.log('CSV data import completed.');

    // Then, start the WebSocket
    startWebSocket();
    console.log('WebSocket connection started.');

    // Catch-all route handler
    app.get("/", (req, res) => {
        res.send("Server is Running!");
    });

    // Router Handlers
    try {
        client.connect();
    } catch (error) {
        console.error("Unable to connect to database.", error);
        process.exit(1);
    };

    // Close the database connection when the server stops
    process.on('exit', () => {
        console.log('Closing database connection');
        client.end();
    });

    // Start the server listening
    const PORT = process.env.PORT || 3001;
    if (process.env.NODE_ENV !== 'test') {
        app.listen(PORT, () => {
            console.log(`Now running on port ${PORT}`);
        });
    }
}

// Start the server
startServer();

// Export
module.exports = {
    app,
    client,
};