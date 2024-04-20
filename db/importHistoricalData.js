require('dotenv').config();
const fs = require('fs');
const csv = require('fast-csv');
const path = require('path');
const { Pool } = require('pg');

// You could use the same client if you are exporting it from './index'
const { client } = require('./index');

const csvDirectory = './db/csvHistory';

// CRUD function to insert historical SPX record
async function createHistoricalSPXRecord({ timestamp, open, high, low, close, volume }) {
    try {
        const { rows: [record] } = await client.query(`
            INSERT INTO historical_spx(timestamp, open, high, low, close, volume)
            VALUES($1, $2, $3, $4, $5, $6)
            RETURNING *;
        `, [timestamp, open, high, low, close, volume]);

        return record;
    } catch (error) {
        console.error("Could not insert historical SPX record.");
        console.log(error);
        throw error;
    }
}

async function getAllHistoricalSPXRecords() {
    try {
        const { rows } = await client.query('SELECT * FROM historical_spx ORDER BY timestamp ASC;');
        return rows;
    } catch (error) {
        console.error("Could not retrieve historical SPX records.", error);
        throw error;
    }
}

async function getHistoricalSPXRecordById(id) {
    try {
        const { rows: [record] } = await client.query('SELECT * FROM historical_spx WHERE id = $1;', [id]);
        return record;
    } catch (error) {
        console.error("Could not retrieve historical SPX record.", error);
        throw error;
    }
}

async function updateHistoricalSPXRecord(id, { open, high, low, close, volume }) {
    try {
        const { rows: [updatedRecord] } = await client.query(`
            UPDATE historical_spx SET open = $2, high = $3, low = $4, close = $5, volume = $6
            WHERE id = $1 RETURNING *;
        `, [id, open, high, low, close, volume]);
        return updatedRecord;
    } catch (error) {
        console.error("Could not update historical SPX record.", error);
        throw error;
    }
}

async function deleteHistoricalSPXRecord(id) {
    try {
        await client.query('DELETE FROM historical_spx WHERE id = $1;', [id]);
        console.log(`Record with id ${id} deleted successfully.`);
    } catch (error) {
        console.error("Could not delete historical SPX record.", error);
        throw error;
    }
}


// Helper function to import a single CSV file
const importCSVFile = async (filePath) => {
    const stream = fs.createReadStream(filePath);

    const csvStream = csv.parse({ headers: true })
        .on('data', async (row) => {
            const [month, day, year] = row['Date'].split('/');
            const formattedDate = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;

            const recordData = {
                timestamp: formattedDate,
                open: parseFloat(row['Open']),
                high: parseFloat(row['High']),
                low: parseFloat(row['Low']),
                close: parseFloat(row['Close/Last']),
                volume: row['Volume'] ? parseInt(row['Volume'], 10) : null
            };

            // Use the CRUD function to insert the data
            const insertedRecord = await createHistoricalSPXRecord(recordData);
            console.log('Inserted Record:', insertedRecord);
        })
        .on('end', () => {
            console.log(`Finished importing data from ${path.basename(filePath)}`);
        })
        .on('error', (err) => {
            console.error('Error reading the CSV file:', err);
        });

    stream.pipe(csvStream);
};

// Main function to loop through all CSV files and import them
const importHistoricalData = async () => {
    const files = fs.readdirSync(csvDirectory);
    for (const file of files) {
        if (path.extname(file) === '.csv') {
            console.log(`Importing data from ${file}`);
            await importCSVFile(path.join(csvDirectory, file));
        }
    }
};

module.exports = {
    createHistoricalSPXRecord,
    getAllHistoricalSPXRecords,
    getHistoricalSPXRecordById,
    updateHistoricalSPXRecord,
    deleteHistoricalSPXRecord,
    importHistoricalData 
};