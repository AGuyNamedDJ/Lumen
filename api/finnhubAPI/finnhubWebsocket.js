const WebSocket = require('ws');
require('dotenv').config();

const API_KEY = process.env.FINNHUB_API_KEY;
const SOCKET_URL = `wss://ws.finnhub.io?token=${API_KEY}`;

const handleWebSocket = () => {
    const socket = new WebSocket(SOCKET_URL);

    socket.on('open', () => {
        console.log('WebSocket connection opened');
        // Subscribe to real-time data for $SPX
        socket.send(JSON.stringify({ 'type': 'subscribe', 'symbol': 'SPX' }));
    });

    socket.on('message', (data) => {
        const parsedData = JSON.parse(data);
        if (parsedData.type === 'trade') {
            parsedData.data.forEach(trade => {
                console.log(`Received trade data: ${JSON.stringify(trade)}`);
                // Call FNs from other modules here to process and store the data
            });
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