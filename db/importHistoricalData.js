const fs = require('fs');
const csv = require('fast-csv');
const { Pool } = require('pg');

const pool = new Pool({
    connectionString: process.env.DATABASE_URL || 'postgres://localhost:5432/lumen',
});

async function insertRecords(records) {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        const query = `
            INSERT INTO historical_spx (timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6);
        `;
        for (const record of records) {
            await client.query(query, record);
        }
        await client.query('COMMIT');
        console.log(`Inserted ${records.length} records successfully`);
    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error inserting records:', error);
    } finally {
        client.release();
    }
}

async function importSpecificCSVFile() {
    const filePath = './db/csvHistory/SPXTest.csv';
    console.log(`Starting to import data from ${filePath}`);
    const stream = fs.createReadStream(filePath);
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
            console.log(`Finished importing data from ${filePath}`);
        })
        .on('error', error => {
            console.error('Error reading the CSV file:', error);
        });
}

module.exports = importSpecificCSVFile;
