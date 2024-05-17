// Requires
require('dotenv').config();
const express = require('express');
const morgan = require('morgan');
const cors = require('cors');
const app = express();

// Import project dirs
const { client } = require('./db/index');
const handleWebSocket = require('./api/finnhubAPI/finnhubWebsocket'); 

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(morgan('dev'));
app.use(cors());

// WebSocket Setup
handleWebSocket();

// Catch-all route handler
app.get("/", (req, res) => {
    res.send("Server is Running!")
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

// Port
const PORT = process.env.PORT || 3001
if (process.env.NODE_ENV !== 'test') {
    app.listen(PORT, () => {
        console.log(`Now running on port ${PORT}`)
    });
};

// Export
module.exports = {
    app,
    client,
};