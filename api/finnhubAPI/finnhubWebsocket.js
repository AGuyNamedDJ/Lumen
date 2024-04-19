// finnhubWebsocket.js
const WebSocket = require('ws');

const socket = new WebSocket('wss://ws.finnhub.io?token=YOUR_API_KEY');

socket.on('open', function open() {
    console.log('Connected to Finnhub websocket');
    // Subscribe to a symbol, e.g., AAPL (Apple Inc.)
    socket.send(JSON.stringify({'type':'subscribe', 'symbol': 'AAPL'}));
});

socket.on('message', function incoming(data) {
    console.log('Received data:', data);
});

socket.on('close', function close() {
    console.log('Disconnected from the websocket');
});

module.exports = socket;
