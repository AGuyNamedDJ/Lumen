// Requires
require('dotenv').config();
const express = require('express');
const morgan = require('morgan');
const cors = require('cors');
const app = express();

// Import project dirs
const { client } = require('./db/index');
const handleWebSocket = require('./api/finnhubAPI/finnhubWebsocket');
const importAllCSVFiles = require('./db/fetchS3Data');

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(morgan('dev'));
app.use(cors());

// Function to start WebSocket
async function startWebSocket() {
    try {
        handleWebSocket();
        console.log('WebSocket connection started');
    } catch (error) {
        console.error('Error starting WebSocket:', error);
    }
}

// Function to import CSV data
async function importCSVData() {
    try {
        await importAllCSVFiles();
        console.log('CSV import completed');
    } catch (error) {
        console.error('Error importing CSV data:', error);
    }
}

// Sequentially run CSV import and then WebSocket
async function startServer() {
    await importCSVData(); // Wait for CSV import to complete before starting WebSocket
    await startWebSocket();
    console.log('Server initialization completed');
}

// Import CSV Data and Start WebSocket on Server Start
startServer();

// Catch-all route handler
app.get("/", (req, res) => {
    res.send("Server is Running!");
});

// Router Handlers
client.connect().then(() => {
    console.log('Connected to the database');
}).catch(error => {
    console.error("Unable to connect to database.", error);
    process.exit(1);
});

// Close the database connection when the server stops
process.on('exit', () => {
    console.log('Closing database connection');
    client.end();
});

// Port
const PORT = process.env.PORT || 3001;
if (process.env.NODE_ENV !== 'test') {
    app.listen(PORT, () => {
        console.log(`Now running on port ${PORT}`);
    });
}

// Export
module.exports = {
    app,
    client,
};