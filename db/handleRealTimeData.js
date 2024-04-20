require('dotenv').config();
const { client } = require("./index");
const WebSocket = require('ws');
const { Pool } = require('pg');

const pool = new Pool({
    connectionString: process.env.DATABASE_URL
});

const socket = new WebSocket(`wss://ws.finnhub.io?token=${process.env.FINNHUB_API_KEY}`);

socket.on('open', function open() {
    console.log('WebSocket connection established.');
    socket.send(JSON.stringify({'type':'subscribe', 'symbol': 'SPX'})); 
});

socket.on('message', async function incoming(data) {
    const jsonData = JSON.parse(data);
    try {
        const client = await pool.connect(); 
        try {
            const query = 'INSERT INTO real_time_spx(timestamp, open, high, low, close, volume) VALUES($1, $2, $3, $4, $5, $6)';
            const values = [new Date(jsonData.t), jsonData.p, jsonData.p, jsonData.p, jsonData.p, jsonData.v];
            await client.query(query, values);
            console.log('Inserted real-time data successfully.');
        } catch (err) {
            console.error('Error inserting real-time data:', err);
        } finally {
            client.release(); 
        }
    } catch (err) {
        console.error('Failed to connect to the database:', err);
    }
});

socket.on('close', function close() {
    console.log('Disconnected from WebSocket');
});
