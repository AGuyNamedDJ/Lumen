const AWS = require('aws-sdk');
const { Client } = require('pg');
const csv = require('fast-csv');

require('dotenv').config();

// Configure AWS SDK with environment variables
AWS.config.update({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION
});

const s3 = new AWS.S3();

const pool = new Client({
    connectionString: process.env.DATABASE_URL,
    ssl: {
        rejectUnauthorized: false
    }
});

async function importSpecificCSVFile() {
    const params = {
        Bucket: 'spx-data-bucket', 
        Key: 'HistoricalData_1713558856481.csv' 
    };

    const stream = s3.getObject(params).createReadStream();
    const records = [];

    stream.pipe(csv.parse({ headers: true }))
        .on('data', row => {
            const [month, day, year] = row['Date'].split('/');
            const formattedDate = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
            records.push([
                formattedDate,
                parseFloat(row['Open']),
                parseFloat(row['High']),
                parseFloat(row['Low']),
                parseFloat(row['Close/Last']),
                parseInt(row['Volume'], 10) || null
            ]);

            if (records.length >= 200) {
                insertRecords([...records]);
                records.length = 0;
            }
        })
        .on('end', () => {
            if (records.length > 0) {
                insertRecords(records);
            }
            console.log(`Finished importing data from S3`);
        })
        .on('error', error => {
            console.error('Error reading the CSV file:', error);
        });
}

async function insertRecords(records) {
    await pool.connect();
    try {
        await pool.query('BEGIN');
        const query = `
            INSERT INTO historical_spx (timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6);
        `;
        for (const record of records) {
            await pool.query(query, record);
        }
        await pool.query('COMMIT');
        console.log(`Inserted ${records.length} records successfully`);
    } catch (error) {
        await pool.query('ROLLBACK');
        console.error('Error inserting records:', error);
    } finally {
        await pool.end();
    }
}

module.exports = importSpecificCSVFile;