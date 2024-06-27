// 1) Imports 
require('dotenv').config();
const { Client } = require('pg');

// 2) Establish a client/DB connection
const client = new Client({
    connectionString: process.env.DATABASE_URL || 'postgres://localhost:5432/lumen',
    ssl: process.env.DATABASE_URL ? { rejectUnauthorized: false } : false
});

// 3) Export 
// Export the client
module.exports = { client };