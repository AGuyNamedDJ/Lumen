const WebSocket = require('ws');
require('dotenv').config(); 
const { createRealTimeSPXRecord } = require('../../db/helperFunctions/realTimeSPX');
const Bottleneck = require('bottleneck');

const API_KEY = 'cohasq1r01qrf6b2ivj0cohasq1r01qrf6b2ivjg';
const SOCKET_URL = `wss://ws.finnhub.io?token=${API_KEY}`;

const limiter = new Bottleneck({
    minTime: 50, // Adjust based on rate limits (20 requests per second)
    maxConcurrent: 1
});

const handleWebSocket = () => {
    const socket = new WebSocket(SOCKET_URL);

    socket.on('open', () => {
        console.log('WebSocket connection opened');
        socket.send(JSON.stringify({ 'type': 'subscribe', 'symbol': '^GSPC' })); // Use the correct symbol here
        console.log('Subscribed to ^GSPC trade data');
    });

    socket.on('message', async (data) => {
        const message = data.toString(); // Convert Buffer to string
        const parsedData = JSON.parse(message); // Parse the JSON string
        console.log('Received data:', parsedData); // Log the parsed data for debugging

        if (parsedData.type === 'trade') {
            console.log(`Received trade data: ${JSON.stringify(parsedData.data)}`);
            parsedData.data.forEach(async (trade) => {
                const timestamp = new Date(trade.t);
                const current_price = trade.p;
                const volume = trade.v;
                const conditions = trade.c ? trade.c.join(', ') : null; // Join conditions array to a string
                console.log(`Extracted volume: ${volume}, conditions: ${conditions}`); // Add logging for volume and conditions
                try {
                    await limiter.schedule(() => createRealTimeSPXRecord({ timestamp, current_price, volume, conditions }));
                    console.log(`Stored trade data: ${JSON.stringify({ timestamp, current_price, volume, conditions })}`);
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

module.exports = handleWebSocket;