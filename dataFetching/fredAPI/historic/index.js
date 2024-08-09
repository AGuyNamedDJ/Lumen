const { fetchAverageHourlyEarningsData } = require('./averageHourlyEarningsData');
const { fetchConsumerConfidenceData } = require('./consumerConfidenceData');
const { fetchConsumerSentimentData } = require('./consumerSentimentData');
const { fetchCoreInflationData } = require('./coreInflationData');
const { fetchCPIData } = require('./cpiData');
const { fetchGDPData } = require('./gdpData');
const { fetchIndustrialProductionData } = require('./industrialProductionData');
const { fetchAllInterestRateData } = require('./interestRateData');

async function fetchAllHistoricFredAPIData() {
    await fetchAverageHourlyEarningsData();
    await fetchConsumerConfidenceData();
    await fetchConsumerSentimentData();
    await fetchCoreInflationData();
    await fetchCPIData();
    await fetchGDPData();
    await fetchIndustrialProductionData();
    await fetchAllInterestRateData();
    // Call other fetch functions for different data points
}

module.exports = { fetchAllHistoricFredAPIData };