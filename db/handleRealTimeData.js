// require('dotenv').config();
// const { Pool } = require('pg');
// const WebSocket = require('ws');
// const { createRealTimeSPXRecord } = require('./helperFunctions/realTimeSPX');

// const pool = new Pool({
//     connectionString: process.env.DATABASE_URL
// });

// const socket = new WebSocket(`wss://ws.finnhub.io?token=${process.env.FINNHUB_API_KEY}`);

// socket.on('open', function open() {
//     console.log('WebSocket connection established.');
//     socket.send(JSON.stringify({'type':'subscribe', 'symbol': 'SPX'})); 
// });

// socket.on('message', async function incoming(data) {
//     const jsonData = JSON.parse(data);
//     if (jsonData.type === 'trade') {
//         jsonData.data.forEach(async (trade) => {
//             const timestamp = new Date(trade.t);
//             const current_price = trade.p;
//             try {
//                 await createRealTimeSPXRecord({ timestamp, current_price });
//                 console.log('Inserted real-time data successfully:', { timestamp, current_price });
//             } catch (error) {
//                 console.error('Error inserting real-time data:', error);
//             }
//         });
//     }
// });

// socket.on('close', function close() {
//     console.log('Disconnected from WebSocket');
// });

// socket.on('error', function error(error) {
//     console.error('WebSocket error:', error);
// });