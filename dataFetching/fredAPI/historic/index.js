const { fetchAverageHourlyEarningsData } = require('./averageHourlyEarningsData');
const { fetchConsumerConfidenceData } = require('./consumerConfidenceData');
const { fetchConsumerSentimentData } = require('./consumerSentimentData');
const { fetchCoreInflationData } = require('./coreInflationData');
const { fetchCPIData } = require('./cpiData');
const { fetchGDPData } = require('./gdpData');
const { fetchIndustrialProductionData } = require('./industrialProductionData');
const { fetchAllInterestRateData } = require('./interestRateData');
const {fetchLaborForceParticipationRateData} = require('./laborForceParticipationRateData');
const {fetchNonfarmPayrollEmploymentData} = require('./nonfarmPayrollEmploymentData');
const {fetchPCEData} = require('./pceData')
const {fetchPPIData} = require('./ppiData')
const{fetchUnemploymentRateData} = require('./unemploymentRateData');

async function fetchAllHistoricFredAPIData() {
    await fetchAverageHourlyEarningsData();
    await fetchConsumerConfidenceData();
    await fetchConsumerSentimentData();
    await fetchCoreInflationData();
    await fetchCPIData();
    await fetchGDPData();
    await fetchIndustrialProductionData();
    await fetchAllInterestRateData();
    await fetchLaborForceParticipationRateData();
    await fetchNonfarmPayrollEmploymentData();
    await fetchPCEData();
    await fetchPPIData();
    await fetchUnemploymentRateData();
}

module.exports = { fetchAllHistoricFredAPIData };