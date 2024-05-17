const fs = require('fs');
const csv = require('fast-csv');
const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
    connectionString: process.env.DATABASE_URL || 'postgres://localhost:5432/lumen',
});

async function insertRecords(records) {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');
        const query = `
            INSERT INTO detailed_historical_spx (timestamp, price, volume)
            VALUES ($1, $2, $3);
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

async function importDetailedHistoricalSPX() {
    const filePath = './db/csvHistory/SP.csv';
    console.log(`Starting to import data from ${filePath}`);
    const stream = fs.createReadStream(filePath);
    const records = [];

    stream.pipe(csv.parse({ headers: true }))
        .on('data', row => {
            try {
                const [month, day, year] = row.date.split('/');
                const formattedDate = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
                const time = row.time;

                // Validate and format date and time
                const timestamp = new Date(`${formattedDate}T${time}Z`);
                if (isNaN(timestamp.getTime())) {
                    console.error(`Invalid date/time found: ${row.date} ${row.time}`);
                } else {
                    records.push([
                        timestamp.toISOString(),
                        parseFloat(row.price),
                        parseInt(row.volume, 10) || null
                    ]);

                    if (records.length >= 200) {
                        insertRecords([...records]);
                        records.length = 0;
                    }
                }
            } catch (error) {
                console.error(`Error processing row: ${error.message}`);
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

module.exports = importDetailedHistoricalSPX;
