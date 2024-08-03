const WebSocket = require('ws');
require('dotenv').config();
const moment = require('moment-timezone');
const { createRealTimeSPXRecord } = require('../../db/helperFunctions/realTimeSPX');
const Bottleneck = require('bottleneck');

const API_KEY = process.env.FINNHUB_API_KEY;
const SOCKET_URL = `wss://ws.finnhub.io?token=${API_KEY}`;

const limiter = new Bottleneck({
    minTime: 3000, // 3 seconds between requests
    maxConcurrent: 1
});

let socket;

const handleWebSocket = () => {
    socket = new WebSocket(SOCKET_URL);

    socket.on('open', () => {
        console.log('WebSocket connection opened');
        socket.send(JSON.stringify({ 'type': 'subscribe', 'symbol': 'GSPC' })); 
        // Possible Symbols FXCM:SPX500, ^GSPC SPY
        console.log('Subscribed to ^GSPC trade data');
    });

    socket.on('message', async (data) => {
        const message = data.toString(); // Convert Buffer to string
        const parsedData = JSON.parse(message); // Parse the JSON string
        console.log('Received data:', parsedData); // Log the parsed data for debugging

        if (parsedData.type === 'trade') {
            console.log(`Received trade data: ${JSON.stringify(parsedData.data)}`);
            parsedData.data.forEach(async (trade) => {
                const utcTimestamp = new Date(trade.t);
                const centralTime = moment(utcTimestamp).tz('America/Chicago').format(); // Convert to Central Time
                const current_price = trade.p;
                const volume = trade.v;
                const conditions = trade.c ? trade.c.join(', ') : null; 
                const serverTime = new Date(); // Get the current server time
                console.log(`API timestamp (Central Time): ${centralTime}, Server timestamp: ${serverTime.toISOString()}`);
                console.log(`Extracted volume: ${volume}, conditions: ${conditions}`); 
                try {
                    await limiter.schedule(async () => {
                        console.log(`Scheduling data storage: ${JSON.stringify({ centralTime, current_price, volume, conditions })}`);
                        await createRealTimeSPXRecord({ timestamp: centralTime, current_price, volume, conditions });
                        console.log('Data stored successfully');
                    });
                } catch (error) {
                    console.error('Error storing real-time SPX data:', error);
                }
            });
        } else if (parsedData.type === 'ping') {
            console.log('Received ping message');
        } else {
            console.log('Received unknown message type:', parsedData.type);
        }
    });

    socket.on('close', () => {
        console.log('WebSocket connection closed');
    });

    socket.on('error', (error) => {
        console.error(`WebSocket error: ${error}`);
    });
};

const restartWebSocket = () => {
    if (socket) {
        socket.terminate();
    }
    handleWebSocket();
};

module.exports = { handleWebSocket, restartWebSocket };