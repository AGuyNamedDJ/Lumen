const AWS = require('aws-sdk');
const csv = require('csv-parser');
const { client } = require('./index');
require('dotenv').config();

// Configure AWS SDK
const s3 = new AWS.S3({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_REGION,
});

// Insert records into the DB
async function insertRecords(records) {
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
  }
}

// Fetch and process the CSV file from S3
async function importSpecificCSVFile() {
  const params = {
    Bucket: 'spx-data-bucket', 
    Key: 'HistoricalData_1713558856481.csv',
  };

  console.log(`Starting to import data from S3 bucket ${params.Bucket} with key ${params.Key}`);
  const records = [];
  let totalRecords = 0;
  let processedRecords = 0;

  const s3Stream = s3.getObject(params).createReadStream();
  const csvStream = s3Stream.pipe(csv());

  csvStream.on('data', (row) => {
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

    totalRecords += 1;

    if (records.length >= 1000) { // Batch size
      insertRecords([...records]);
      processedRecords += records.length;
      console.log(`Processed ${processedRecords} of ${totalRecords} records`);
      records.length = 0;
    }
  });

  csvStream.on('end', () => {
    if (records.length > 0) {
      insertRecords(records);
      processedRecords += records.length;
    }
    console.log(`Finished importing data from S3. Processed ${processedRecords} records in total.`);
  });

  csvStream.on('error', (error) => {
    console.error('Error reading the CSV file from S3:', error);
  });
}

module.exports = importSpecificCSVFile;

// Run
// importSpecificCSVFile();