const getCurrentDateTime = () => {
    return new Date().toLocaleString();
};

module.exports = { getCurrentDateTime };