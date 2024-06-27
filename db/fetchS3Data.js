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
            let record;
            // Check for the column names to handle different formats
            if (row['Close/Last']) {
                const [month, day, year] = row['Date'].split('/');
                const formattedDate = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
                record = [
                    formattedDate,
                    parseFloat(removeCommas(row['Open'])),
                    parseFloat(removeCommas(row['High'])),
                    parseFloat(removeCommas(row['Low'])),
                    parseFloat(removeCommas(row['Close/Last'])),
                    parseInt(removeCommas(row['Volume']), 10) || null
                ];
            } else {
                const [month, day, year] = row['Date'].split('/');
                const formattedDate = `${year}-${month.padStart(2, '0')}-${day.padStart(2, '0')}`;
                record = [
                    formattedDate,
                    parseFloat(removeCommas(row['Open'])),
                    parseFloat(removeCommas(row['High'])),
                    parseFloat(removeCommas(row['Low'])),
                    parseFloat(removeCommas(row['Close'])),
                    parseInt(removeCommas(row['Volume']), 10) || null
                ];
            }

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
        'HistoricalData_1713558856481.csv',
        'SPX-Data-2024.csv', 
        'SPX-Data-2013.csv', 
        'SPX-Data-2012.csv',
        'SPX-Data-2011.csv', 
        'SPX-Data-2010.csv', 
        'SPX-Data-2009.csv',
        'SPX-Data-2008.csv', 
        'SPX-Data-2007.csv', 
        'SPX-Data-2006.csv',
        'SPX-Data-2005.csv', 
        'SPX-Data-2004.csv',
        'SPX-Data-2003.csv',
        'SPX-Data-2002.csv',
        'SPX-Data-2001.csv',
        'SPX-Data-2000.csv',
        'SPX-Data-1999.csv',
        'SPX-Data-1998.csv',
        'SPX-Data-1997.csv',
        'SPX-Data-1996.csv',
        'SPX-Data-1995.csv',
        'SPX-Data-1994.csv',
        'SPX-Data-1993.csv',
        'SPX-Data-1992.csv',
        'SPX-Data-1991.csv',
        'SPX-Data-1990.csv',
        'SPX-Data-1989.csv',
        'SPX-Data-1988.csv',
        'SPX-Data-1987.csv',
        'SPX-Data-1986.csv',
        'SPX-Data-1985.csv',
        'SPX-Data-1984.csv',
        'SPX-Data-1983.csv',
        'SPX-Data-1982.csv',
        'SPX-Data-1981.csv',
        'SPX-Data-1980.csv',
        'SPX-Data-1979.csv',
        'SPX-Data-1978.csv'
    ];

    for (const file of files) {
        console.log(`Starting to import data from ${file}`);
        await importSpecificCSVFile(file);
    }

    console.log('Finished importing all CSV files.');
}

module.exports = importAllCSVFiles;