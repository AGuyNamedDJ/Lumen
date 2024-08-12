const { fetchAggregatesData } = require('../dataFetching/indicators/fetch/aggregates');
const { fetchEMAData } = require('../dataFetching/indicators/fetch/emaIndicators');
const { fetchMACDData } = require('../dataFetching/indicators/fetch/macdIndicators');
const { fetchAndStoreRSIData } = require('../dataFetching/indicators/fetch/rsiIndicators');
const { fetchAndStoreSMAData } = require('../dataFetching/indicators/fetch/smaIndicators');

async function testIndicators() {
    const symbol = 'SPY';

    // console.log('Testing Aggregates Data Fetch...');
    // await fetchAggregatesData(symbol, 1, 'day', 'YYYY-MM-DD', 'YYYY-MM-DD');

    console.log('Testing EMA Data Fetch...');
    await fetchEMAData(symbol, 20, 'close');

    console.log('Testing MACD Data Fetch...');
    await fetchMACDData(symbol, 'day', 'close', 12, 26, 9);

    console.log('Testing RSI Data Fetch...');
    await fetchAndStoreRSIData(symbol, 14, 'close');

    console.log('Testing SMA Data Fetch...');
    await fetchAndStoreSMAData(symbol, 20, 'close');

    console.log('All indicators have been tested.');
}

testIndicators().catch(error => console.error('Error during testing:', error));