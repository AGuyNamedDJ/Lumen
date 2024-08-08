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
        Bucket: 'spx-data-bucket', // Name of the S3 bucket
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
async function importAllSPXCSVFiles() {
    const files = [
        'SPX-1980.csv', 'SPX-1990.csv',
        'SPX-1995.csv', 'SPX-1996.csv',
        'SPX-1997.csv', 'SPX-1998.csv',
        'SPX-1999.csv', 'SPX-2000.csv',
        'SPX-2001.csv', 'SPX-2002.csv',
        'SPX-2003.csv', 'SPX-2004.csv',
        'SPX-2005.csv', 'SPX-2006.csv',
        'SPX-2007.csv', 'SPX-2008.csv',
        'SPX-2009.csv', 'SPX-2010.csv',
        'SPX-2011.csv', 'SPX-2012.csv',
        'SPX-2013.csv', 'SPX-2014.csv',
        'SPX-2015.csv', 'SPX-2016.csv',
        'SPX-2017.csv', 'SPX-2018.csv',
        'SPX-2019.csv', 'SPX-2020.csv',
        'SPX-2021.csv', 'SPX-2022.csv',
        'SPX-2023.csv', 'SPX-2024.csv',
        'SPX-2024-8.csv',
    ];

    for (const file of files) {
        console.log(`Starting to import data from ${file}`);
        await importSpecificCSVFile(file);
    }

    console.log('Finished importing all CSV files.');
}

module.exports = importAllSPXCSVFiles;