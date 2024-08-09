const { fetchAverageHourlyEarningsData } = require('./averageHourlyEarningsData');
const { fetchConsumerConfidenceData } = require('./consumerConfidenceData');
const { fetchConsumerSentimentData } = require('./consumerSentimentData');
const { fetchCoreInflationData } = require('./coreInflationData');
const { fetchCPIData } = require('./cpiData');


async function fetchAllHistoricFredAPIData() {
    await fetchAverageHourlyEarningsData();
    await fetchConsumerConfidenceData();
    await fetchConsumerSentimentData();
    await fetchCoreInflationData();
    await fetchCPIData();
    // Call other fetch functions for different data points
}

module.exports = { fetchAllHistoricFredAPIData };