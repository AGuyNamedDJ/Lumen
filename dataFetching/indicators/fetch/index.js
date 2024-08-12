// const { fetchAggregatesData } = require('./aggregates');
const { fetchEMAData } = require('./emaIndicators')
const { fetchMACDData } = require('./macdIndicators');
const { fetchAndStoreRSIData } = require('./rsiIndicators');
const { fetchAndStoreSMAData } = require('./smaIndicators');
async function fetchAllIndicatorsData() {
    // await fetchAggregatesData();
    await fetchEMAData();
    await fetchMACDData();
    await fetchAndStoreRSIData();
    await fetchAndStoreSMAData();
};

module.exports = { fetchAllIndicatorsData };