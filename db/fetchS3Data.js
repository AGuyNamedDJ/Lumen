const AWS = require('aws-sdk');
const { client } = require('./index');
const csv = require('fast-csv');
require('dotenv').config();

// Configure AWS SDK with environment variables
AWS.config.update({
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    region: process.env.AWS_REGION
});

const s3 = new AWS.S3();

/**
 * Function to remove commas from a string
 * @param {string} str - The string to remove commas from
 * @returns {string} - The string without commas
 */
const removeCommas = str => str ? str.replace(/,/g, '') : '0';

/**
 * Function to import data from a specific CSV file stored in an S3 bucket
 * @param {string} fileKey - The key of the CSV file in the S3 bucket
 */
async function importSpecificCSVFile(fileKey) {
    const params = {
        Bucket: 'spy-data-bucket', // Name of the S3 bucket
        Key: fileKey // Key of the CSV file in the S3 bucket
    };

    const stream = s3.getObject(params).createReadStream();
    const records = [];
    let totalRecords = 0;
    let totalInserted = 0;

    // Parse the CSV file
    stream.pipe(csv.parse({ headers: true }))
        .on('data', row => {
            const formattedDate = row['Date'];
            const record = [
                formattedDate,
                parseFloat(removeCommas(row['Open'])),
                parseFloat(removeCommas(row['High'])),
                parseFloat(removeCommas(row['Low'])),
                parseFloat(removeCommas(row['Close'])),
                parseInt(removeCommas(row['Volume']), 10) || null
            ];

            records.push(record);
            totalRecords++;
            if (records.length >= 1000) {
                insertRecords([...records], totalRecords, totalInserted);
                totalInserted += records.length;
                records.length = 0;
            }
        })
        .on('end', async () => {
            if (records.length > 0) {
                await insertRecords(records, totalRecords, totalInserted);
            }
            console.log(`Finished importing data from ${fileKey}. Total records processed: ${totalRecords}.`);
        })
        .on('error', error => {
            console.error('Error reading the CSV file:', error);
        });
}

/**
 * Function to insert records into the database
 * @param {Array} records - The records to be inserted
 * @param {number} totalRecords - The total number of records read so far
 * @param {number} totalInserted - The total number of records inserted so far
 */
async function insertRecords(records, totalRecords, totalInserted) {
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
        totalInserted += records.length;
        const totalRemaining = totalRecords - totalInserted;
        console.log(`Inserted ${records.length} records successfully. Total records remaining: ${totalRemaining}`);
    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error inserting records:', error);
    }
}

/**
 * Function to import all CSV files from S3
 */
async function importAllCSVFiles() {
    const files = [
        'SPY.csv'
        // Add other CSV files as needed
    ];

    for (const file of files) {
        console.log(`Starting to import data from ${file}`);
        await importSpecificCSVFile(file);
    }

    console.log('Finished importing all CSV files.');
}

module.exports = importAllCSVFiles;