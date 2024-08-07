// Import Client & Exports;
const { client } = require('./index');

// File Imports
const importSpecificCSVFile = require('./importHistoricalData');
const importDetailedHistoricalSPX = require('./importDetailedHistoricalData');
const { createUser, getAllUsers, getUserById, getUserByUsername, deleteUser, updateUser, loginUser } = require('./helperFunctions/user');
const { createHistoricalRecord, getAllHistoricalRecords, getHistoricalRecordById, updateHistoricalRecord, deleteHistoricalRecord } = require ('./helperFunctions/historicalSPX');
const { createStrategy, getAllStrategies, getStrategyByName, getStrategyById, updateStrategy, deleteStrategy } = require('./helperFunctions/strategies');
const { createTrade, getAllTrades, getTradeById, updateTrade, deleteTrade } = require('./helperFunctions/trades');
const { createDecisionRule, getAllDecisionRules, getDecisionRuleById, updateDecisionRule, deleteDecisionRule } = require('./helperFunctions/decisionrules');
const { createAlert, getAllAlerts, getAlertById, updateAlert, deleteAlert} = require('./helperFunctions/alerts');
const { createAuditLog, getAllAuditLogs, getAuditLogById, deleteAuditLog } = require('./helperFunctions/auditLogs');
const { createRealTimeSPXRecord, getAllRealTimeSPXRecords, getRealTimeSPXRecordById, updateRealTimeSPXRecord, deleteRealTimeSPXRecord } = require('./helperFunctions/realTimeSPX');
const { createDetailedRecord, getAllDetailedRecords, getDetailedRecordById, updateDetailedRecord, deleteDetailedRecord} = require('./helperFunctions/detailedHistoricalSPX');
const { createConversation, getAllConversations, getConversationById, getConversationsByUserId, updateConversation, deleteConversation } = require('./helperFunctions/conversations');
const { createMessage, getAllMessages, getMessageById, getMessagesByConversationId, updateMessage, deleteMessage} = require('./helperFunctions/messages');
const { storeAggregatesData, getAllAggregates, getAggregatesBySymbol, getAggregatesBySymbolAndTimespan, deleteAggregates} = require('./indicators/aggregates');
// const { storeBBData, getAllBBIndicators, getBBIndicatorById, getBBIndicatorsBySymbol, deleteBBIndicator } = require('./indicators/bbindicators');
const { storeEMAData, getAllEMAIndicators, getEMAIndicatorById, getEMAIndicatorsBySymbol, deleteEMAIndicator} = require('./indicators/emaIndicators');
const { storeMACDData, getAllMACDIndicators, getMACDIndicatorById, getMACDIndicatorsBySymbol, deleteMACDIndicator} = require('./indicators/macdIndicators');
const { storeRSIData, getAllRSIIndicators, getRSIIndicatorById, getRSIIndicatorsBySymbol, deleteRSIIndicator} = require('./indicators/rsiIndicators');
const { storeSMAData, getAllSMAIndicators, getSMAIndicatorById, getSMAIndicatorsBySymbol, deleteSMAIndicator} = require('./indicators/smaIndicators');
const { storeCPIData, getAllCPIData, getCPIDataByDate, updateCPIDataByDate, deleteCPIDataByDate,} = require('./fredAPI/cpiData')
const { storePPIData, getAllPPIData, getPPIDataByDate, updatePPIDataByDate, deletePPIDataByDate,} = require ('./fredAPI/ppiData');
const { storeUnemploymentRateData, getAllUnemploymentRateData, getUnemploymentRateDataByDate, updateUnemploymentRateDataByDate, deleteUnemploymentRateDataByDate} = require ('./fredAPI/unemploymentRateData');


// Methods: Drop Tables
async function dropTables() {
    try {
        console.log("Dropping tables except real_time_spx...");
        await client.query(`
            DROP TABLE IF EXISTS users CASCADE;
            DROP TABLE IF EXISTS strategies CASCADE;
            DROP TABLE IF EXISTS trades CASCADE;
            DROP TABLE IF EXISTS aggregates CASCADE;
            DROP TABLE IF EXISTS bb_indicators CASCADE;
            DROP TABLE IF EXISTS ema_indicators CASCADE;
            DROP TABLE IF EXISTS macd_indicators CASCADE;
            DROP TABLE IF EXISTS rsi_indicators CASCADE;
            DROP TABLE IF EXISTS sma_indicators CASCADE;
            DROP TABLE IF EXISTS cpi_data CASCADE;
            DROP TABLE IF EXISTS ppi_data CASCADE;
            DROP TABLE IF EXISTS unemployment_rate_data CASCADE;
            DROP TABLE IF EXISTS decision_rules;
            DROP TABLE IF EXISTS alerts CASCADE;
            DROP TABLE IF EXISTS audit_logs CASCADE;
            DROP TABLE IF EXISTS conversations CASCADE;
            DROP TABLE IF EXISTS messages CASCADE;
            DROP TABLE IF EXISTS historical_spx CASCADE;
            DROP TABLE IF EXISTS historical_spy CASCADE;
            DROP TABLE IF EXISTS detailed_historical_spx CASCADE;
            DROP TABLE IF EXISTS detailed_historical_spy CASCADE;
            -- Note: Do not drop real_time_spx or real_time_spy table
        `);
        console.log("Finished dropping tables.")
    } catch (error) {
        console.log("Error dropping tables!")
        console.log(error)
    }
};

// Methods: Create Tables
async function createTables() {
    try {
        console.log('Starting to build tables...');
        await client.query(`
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            phone_number VARCHAR(15),
            date_of_birth DATE,
            role VARCHAR(50) DEFAULT 'user',
            status VARCHAR(50) DEFAULT 'active',
            profile_picture_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(50) NOT NULL, -- 'user' or 'ai'
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS historical_spx (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume BIGINT
        );
        CREATE TABLE IF NOT EXISTS historical_spy (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume BIGINT
        );
        CREATE TABLE IF NOT EXISTS aggregates (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            multiplier INT NOT NULL,
            timespan TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, multiplier, timespan, timestamp)
        );
        CREATE TABLE IF NOT EXISTS bb_indicators (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            period INT NOT NULL,
            timespan TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            middle_band FLOAT NOT NULL,
            upper_band FLOAT NOT NULL,
            lower_band FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, period, timespan, timestamp)
        );
        CREATE TABLE IF NOT EXISTS ema_indicators (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            period INT NOT NULL,
            timespan TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, period, timespan, timestamp)
        );
        CREATE TABLE IF NOT EXISTS macd_indicators (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            period INT NOT NULL,
            timespan TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            macd_line FLOAT NOT NULL,
            signal_line FLOAT NOT NULL,
            histogram FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, period, timespan, timestamp)
        );
        CREATE TABLE IF NOT EXISTS rsi_indicators (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            period INT NOT NULL,
            timespan TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, period, timespan, timestamp)
        );
        CREATE TABLE IF NOT EXISTS sma_indicators (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            period INT NOT NULL,
            timespan TEXT NOT NULL,
            timestamp BIGINT NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, period, timespan, timestamp)
        );
        CREATE TABLE IF NOT EXISTS cpi_data (
            id SERIAL PRIMARY KEY,
            date DATE UNIQUE NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS ppi_data (
            id SERIAL PRIMARY KEY,
            date DATE UNIQUE NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS unemployment_rate_data (
            id SERIAL PRIMARY KEY,
            date DATE UNIQUE NOT NULL,
            value FLOAT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS strategies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id),
            open_time TIMESTAMP NOT NULL,
            close_time TIMESTAMP,
            status VARCHAR(50) NOT NULL,
            entry_price NUMERIC,
            exit_price NUMERIC,
            profit_loss NUMERIC
        );
        CREATE TABLE IF NOT EXISTS decision_rules (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id),
            parameter_name VARCHAR(255) NOT NULL,
            value TEXT NOT NULL,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            trade_id INTEGER REFERENCES trades(id),
            strategy_id INTEGER REFERENCES strategies(id),
            message TEXT NOT NULL,
            alert_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            action_type VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS real_time_spx (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            current_price NUMERIC NOT NULL,
            volume BIGINT,
            conditions TEXT,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC
        );
        CREATE TABLE IF NOT EXISTS real_time_spy (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            current_price NUMERIC NOT NULL,
            volume BIGINT,
            conditions TEXT,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC
        );
        CREATE TABLE IF NOT EXISTS detailed_historical_spx (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            price NUMERIC NOT NULL,
            volume BIGINT
        );
        CREATE TABLE IF NOT EXISTS detailed_historical_spy (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            price NUMERIC NOT NULL,
            volume BIGINT
        );
    `);
    console.log('Finished building tables.');

    // Print all the tables
    const tables = await client.query(`
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    `);
    console.log('Tables in the database:');
    tables.rows.forEach(row => console.log(row.table_name));

    } catch (error) {
        console.error('Error building tables!');
        console.log(error);
    }
}

// createInitialUsers
async function createInitialUsers() {
    console.log("Creating initial users...");
    try {
        await createUser({
            username: 'AGNDJ', 
            password: 'Dalron123!', 
            email: 'dalronj.robertson@gmail.com', 
            first_name: 'Dalron J.', 
            last_name: 'Robertson', 
            phone_number: '601-720-0154',
            date_of_birth: '1995-08-08',
            profile_picture_url: 'https://dalronjrobertson.com/images/AGNDJ.png',
            role: 'admin'
        });

        console.log("Finished creating initial users.");
    } catch (error) {
        console.error("Error creating initial users!");
        console.error(error);
    }
};

// createInitialConversations
async function createInitialConversations() {
    console.log("Creating initial conversations...");
    try {
        const conversation = await createConversation({
            userId: 1,
        });

        await createMessage({
            conversationId: conversation.id,
            role: 'user',
            content: 'Hello, this is the initial message from the user.'
        });

        await createMessage({
            conversationId: conversation.id,
            role: 'ai',
            content: 'Hello, this is the initial message from the AI.'
        });

        console.log("Finished creating initial conversations.");
    } catch (error) {
        console.error("Error creating initial conversations!");
        console.error(error);
    }
};
// Function to check if the table exists
async function checkTableExists(tableName) {
    try {
        const res = await client.query(`
            SELECT EXISTS (
                SELECT FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename = $1
            );
        `, [tableName]);
        return res.rows[0].exists;
    } catch (error) {
        console.error(`Error checking if table ${tableName} exists:`, error);
        throw error;
    }
};

// Function to create the CPI table if it does not exist
async function createCPITableIfNotExists() {
    try {
        const tableExists = await checkTableExists('cpi_data');
        if (!tableExists) {
            await client.query(`
                CREATE TABLE IF NOT EXISTS cpi_data (
                    id SERIAL PRIMARY KEY,
                    date DATE UNIQUE NOT NULL,
                    value FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            `);
            console.log('cpi_data table created.');
        } else {
            console.log('cpi_data table already exists.');
        }
    } catch (error) {
        console.error('Error creating cpi_data table:', error);
        throw error;
    }
};

// createInitialCPIData
async function createInitialCPIData() {
    console.log("Creating initial CPI data...");
    try {
        const cpiDataEntries = [
            { date: '2021-01-01', value: 260.474 }
            // Add more CPI data as needed
        ];

        for (const entry of cpiDataEntries) {
            console.log(`Storing CPI Data: Date - ${entry.date}, Value - ${entry.value}`);
            await storeCPIData(entry);
        }

        console.log("Finished creating initial CPI data.");
    } catch (error) {
        console.error("Error creating initial CPI data!");
        console.error(error);
    }
};

// // createInitialMessages
// async function createInitialMessages(conversationId) {
//     console.log("Creating initial messages...");
//     try {
//         const conversation = await getConversationById(conversationId);
//         if (!conversation) {
//             throw new Error('Conversation not found');
//         }

//         // Create initial messages for the conversation
//         await createMessage({
//             conversationId: conversation.id,
//             role: 'user',
//             content: 'Hello, this is the initial message from the user.'
//         });

//         await createMessage({
//             conversationId: conversation.id,
//             role: 'ai',
//             content: 'Hello, this is the initial message from the AI.'
//         });

//         console.log("Finished creating initial messages.");
//     } catch (error) {
//         console.error("Error creating initial messages!");
//         console.error(error);
//     }
// };

// createInitialHistoricalSPX
// async function createInitialHistoricalSPX() {
//     console.log("Creating initial historical SPX data...");
//     try {
//         const historicalRecords = [
//             { timestamp: new Date('2023-01-01'), open: 4200.0, high: 4250.0, low: 4150.0, close: 4225.0, volume: 500000 },
//             { timestamp: new Date('2023-01-02'), open: 4225.0, high: 4300.0, low: 4200.0, close: 4280.0, volume: 600000 },
//             // Add more records as needed
//         ];

//         for (const record of historicalRecords) {
//             await createHistoricalRecord(record);
//         }

//         console.log("Finished creating initial historical SPX data.");
//     } catch (error) {
//         console.error("Error creating initial historical SPX data!");
//         console.error(error);
//     }
// };


// createInitialStrategies
async function createInitialStrategies() {
    console.log("Creating initial strategies...");
    try {
        const strategies = [
            { name: 'Buy Stock', description: 'Buying the stock.' },
            { name: 'Sell Stock', description: 'Selling the stock.' },
            { name: 'Buy Call', description: 'Buying a call option.' },
            { name: 'Sell Call', description: 'Selling a call option.' },
            { name: 'Buy Put', description: 'Buying a put option.' },
            { name: 'Sell Put', description: 'Selling a put option.' },
            { name: 'Bull Call Spread', description: 'A vertical spread strategy where you buy a call and sell a higher strike call.' },
            { name: 'Bear Put Spread', description: 'A vertical spread strategy where you buy a put and sell a lower strike put.' },
            { name: 'Iron Condor', description: 'An options trading strategy that involves buying and selling four options contracts with different strike prices.' },
        ];

        for (const strategy of strategies) {
            const existingStrategy = await getStrategyByName(strategy.name);
            if (!existingStrategy) {
                await createStrategy(strategy);
            }
        }
        console.log("Finished creating initial strategies.");
    } catch (error) {
        console.error("Error creating initial strategies!");
        console.error(error);
    }
};

// createInitialTrades
async function createInitialTrades() {
    console.log("Creating initial trades...");
    try {
        const trades = [
            {
                strategy_id: 1,  // Assuming strategy_id 1 exists
                open_time: new Date('2023-01-01T10:00:00Z'),
                close_time: new Date('2023-01-01T16:00:00Z'),
                status: 'closed',
                entry_price: 4200.0,
                exit_price: 4250.0,
                profit_loss: 50.0
            },
            {
                strategy_id: 2,  // Assuming strategy_id 2 exists
                open_time: new Date('2023-01-02T10:00:00Z'),
                close_time: null,  // Trade still open
                status: 'open',
                entry_price: 4225.0,
                exit_price: null,
                profit_loss: null
            },

        ];

        for (const trade of trades) {
            await createTrade(trade);
        }

        console.log("Finished creating initial trades.");
    } catch (error) {
        console.error("Error creating initial trades!");
        console.error(error);
    }
};

// createInitialDecisionRules
async function createInitialDecisionRules() {
    console.log("Creating initial decision rules...");
    try {
        const decisionRules = [
            {
                strategy_id: 1,  // Assuming strategy_id 1 exists
                parameter_name: 'entry_signal',
                value: 'RSI < 30',
                description: 'Enter a trade when RSI is below 30'
            },
            {
                strategy_id: 2,  // Assuming strategy_id 2 exists
                parameter_name: 'exit_signal',
                value: 'RSI > 70',
                description: 'Exit a trade when RSI is above 70'
            },
            // Add more decision rules as needed
        ];

        for (const rule of decisionRules) {
            await createDecisionRule(rule);
        }

        console.log("Finished creating initial decision rules.");
    } catch (error) {
        console.error("Error creating initial decision rules!");
        console.error(error);
    }
};

// createInitialAlerts
async function createInitialAlerts() {
    console.log("Creating initial alerts...");
    try {
        const alerts = [
            {
                user_id: 1,
                trade_id: 1,  
                strategy_id: 1,  
                message: 'Trade is available to take',
                alert_type: 'trade_available'
            },
            {
                user_id: 1,  
                trade_id: 2,  
                strategy_id: 2,  
                message: 'Trade has hit optimal profit',
                alert_type: 'optimal_profit'
            },
            // Add more alerts as needed
        ];

        for (const alert of alerts) {
            await createAlert(alert);
        }

        console.log("Finished creating initial alerts.");
    } catch (error) {
        console.error("Error creating initial alerts!");
        console.error(error);
    }
};

// createInitialAuditLogs
async function createInitialAuditLogs() {
    console.log("Creating initial audit logs...");
    try {
        const auditLogs = [
            {
                action_type: 'login',
                description: 'User admin logged in'
            },
            {
                action_type: 'trade_execution',
                description: 'Executed trade for strategy ID 1'
            },
            // Add more audit logs as needed
        ];

        for (const log of auditLogs) {
            await createAuditLog(log);
        }

        console.log("Finished creating initial audit logs.");
    } catch (error) {
        console.error("Error creating initial audit logs!");
        console.error(error);
    }
};

// createInitialRealTimeSPXData
// async function createInitialRealTimeSPXData() {
//     console.log("Creating initial real-time SPX data...");
//     try {
//         const realTimeData = [
//             {
//                 timestamp: new Date('2024-05-17T14:00:00Z'),
//                 current_price: 5286.0
//             },
//             {
//                 timestamp: new Date('2024-05-17T14:01:00Z'),
//                 current_price: 5287.5
//             },
//             // Add more records as needed
//         ];

//         for (const data of realTimeData) {
//             await createRealTimeSPXRecord(data);
//         }

//         console.log("Finished creating initial real-time SPX data.");
//     } catch (error) {
//         console.error("Error creating initial real-time SPX data!");
//         console.error(error);
//     }
// };

// // createInitialDetailedHistoricalSPX
// async function createInitialDetailedHistoricalSPX() {
//     console.log("Creating initial detailed historical SPX data...");
//     try {
//         const data = [
//             { timestamp: new Date('2024-01-01T10:00:00Z'), price: 4300.12, volume: 12345 },
//             { timestamp: new Date('2024-01-01T10:01:00Z'), price: 4301.45, volume: 6789 },
//             // Add more initial data as needed
//         ];

//         for (const record of data) {
//             await createDetailedRecord(record);
//         }

//         console.log("Finished creating initial detailed historical SPX data.");
//     } catch (error) {
//         console.error("Error creating initial detailed historical SPX data!");
//         console.error(error);
//     }
// };

// Rebuild DB
async function rebuildDB() {
    try {
        await client.connect();
        await dropTables();
        await createTables();
        await createInitialUsers();
        await createInitialConversations();
        await checkTableExists();
        await createCPITableIfNotExists();
        await createInitialCPIData();
        // await createInitialHistoricalSPX();
        await createInitialStrategies();
        await createInitialTrades();
        await createInitialDecisionRules();
        await createInitialAlerts();
        await createInitialAuditLogs();
        // await createInitialRealTimeSPXData();
        // await createInitialDetailedHistoricalSPX();
        console.log('Tables have been successfully created.');
    } catch (error) {
        console.error("Error during rebuildDB!");
        console.log(error.detail);
    }
};

// Test Suite
async function testDB() {
    try {
        console.log("Starting to test database...");

        // // Test User Helper FNs
        // console.log("Starting to test users...");
        // const initialUser = await getUserByUsername('Admin');
        // console.log("Initial user", initialUser);

        // if (initialUser) {
        //     // Test getAllUsers
        //     console.log("Calling getAllUsers...");
        //     const allUsers = await getAllUsers();
        //     console.log("All users", allUsers);

        //     // Test getUserById
        //     console.log("Calling getUserById for the initial user...");
        //     const userById = await getUserById(initialUser.id);
        //     console.log("User by ID", userById);

        //     // Test updateUser
        //     console.log("Updating initial user's last name...");
        //     const updatedUser = await updateUser(initialUser.username, { last_name: 'UpdatedLastName' });
        //     console.log("Updated user", updatedUser);

        //     // Test deleteUser
        //     console.log("Deleting the initial user...");
        //     const deletedUser = await deleteUser(initialUser.username);
        //     console.log("Deleted user", deletedUser);
        // };

        // console.log("Creating initial user...");
        // await createInitialUsers();
        // const initialUser = await getUserByUsername('Owner1');
        // console.log("Initial user", initialUser);

        // if (initialUser) {
        //     // Test createConversation
        //     console.log("Creating a new conversation...");
        //     const newConversation = await createConversation({ userId: initialUser.id });
        //     console.log("New conversation", newConversation);

        //     // Test getAllConversations
        //     console.log("Calling getAllConversations...");
        //     const allConversations = await getAllConversations();
        //     console.log("All conversations", allConversations);

        //     // Test getConversationById
        //     console.log("Calling getConversationById for the new conversation...");
        //     const conversationById = await getConversationById(newConversation.id);
        //     console.log("Conversation by ID", conversationById);

        //     // Test updateConversation
        //     console.log("Updating new conversation's updated_at timestamp...");
        //     const updatedConversation = await updateConversation(newConversation.id, { updated_at: new Date() });
        //     console.log("Updated conversation", updatedConversation);

        //     // Test createMessage
        //     console.log("Creating messages for the conversation...");
        //     const userMessage = await createMessage({ conversationId: newConversation.id, role: 'user', content: 'Hello, this is the initial message from the user.' });
        //     console.log("User message", userMessage);

        //     const aiMessage = await createMessage({ conversationId: newConversation.id, role: 'ai', content: 'Hello, this is the initial message from the AI.' });
        //     console.log("AI message", aiMessage);

        //     // Test getAllMessages
        //     console.log("Calling getAllMessages...");
        //     const allMessages = await getAllMessages();
        //     console.log("All messages", allMessages);

        //     // Test getMessageById
        //     console.log("Calling getMessageById for the user message...");
        //     const messageById = await getMessageById(userMessage.id);
        //     console.log("Message by ID", messageById);

        //     // Test getMessagesByConversationId
        //     console.log("Calling getMessagesByConversationId for the new conversation...");
        //     const messagesByConversationId = await getMessagesByConversationId(newConversation.id);
        //     console.log("Messages by conversation ID", messagesByConversationId);

        //     // Test updateMessage
        //     console.log("Updating user message content...");
        //     const updatedMessage = await updateMessage(userMessage.id, { content: 'Updated user message content.' });
        //     console.log("Updated message", updatedMessage);

        //     // Test deleteMessage
        //     console.log("Deleting the AI message...");
        //     const deletedMessage = await deleteMessage(aiMessage.id);
        //     console.log("Deleted message", deletedMessage);

        //     // Test deleteConversation
        //     console.log("Deleting the new conversation...");
        //     const deletedConversation = await deleteConversation(newConversation.id);
        //     console.log("Deleted conversation", deletedConversation);
        // }

        // // Test Historical SPX Helper FNs
        // console.log("Starting to test historical SPX...");

        // // Get all historical SPX records
        // console.log("Calling getAllHistoricalRecords...");
        // const allHistoricalRecords = await getAllHistoricalRecords();
        // console.log("All historical SPX records", allHistoricalRecords);

        // // Assuming at least one historical SPX record is created successfully
        // if (allHistoricalRecords.length > 0) {
        //     // Get historical SPX record by ID
        //     console.log("Calling getHistoricalRecordById for the first record...");
        //     const historicalRecordById = await getHistoricalRecordById(allHistoricalRecords[0].id);
        //     console.log("Historical SPX record by ID", historicalRecordById);

        //     // Update historical SPX record
        //     console.log("Updating first historical SPX record's volume...");
        //     const updatedHistoricalRecord = await updateHistoricalRecord(allHistoricalRecords[0].id, { volume: 700000 });
        //     console.log("Updated historical SPX record", updatedHistoricalRecord);

        //     // Delete historical SPX record
        //     console.log("Deleting the first historical SPX record...");
        //     const deletedHistoricalRecord = await deleteHistoricalRecord(allHistoricalRecords[0].id);
        //     console.log("Deleted historical SPX record", deletedHistoricalRecord);
        // };

        // // Test Strategies Helper FNs
        // console.log("Starting to test strategies...");
        // await createInitialStrategies();

        // // Get all strategies
        // console.log("Calling getAllStrategies...");
        // const allStrategies = await getAllStrategies();
        // console.log("All strategies", allStrategies);

        // // Assuming at least one strategy is created successfully
        // if (allStrategies.length > 0) {
        //     // Get strategy by ID
        //     console.log("Calling getStrategyById for the first strategy...");
        //     const strategyById = await getStrategyById(allStrategies[0].id);
        //     console.log("Strategy by ID", strategyById);

        //     // Update strategy
        //     console.log("Updating first strategy's description...");
        //     const updatedStrategy = await updateStrategy(allStrategies[0].id, { description: 'Updated strategy description' });
        //     console.log("Updated strategy", updatedStrategy);

        //     // Delete strategy
        //     console.log("Deleting the first strategy...");
        //     const deletedStrategy = await deleteStrategy(allStrategies[0].id);
        //     console.log("Deleted strategy", deletedStrategy);
        // };

        // // Test Trades Helper FNs
        // console.log("Starting to test trades...");

        // // Get all trades
        // console.log("Calling getAllTrades...");
        // const allTrades = await getAllTrades();
        // console.log("All trades", allTrades);

        // // Assuming at least one trade is created successfully
        // if (allTrades.length > 0) {
        //     // Get trade by ID
        //     console.log("Calling getTradeById for the first trade...");
        //     const tradeById = await getTradeById(allTrades[0].id);
        //     console.log("Trade by ID", tradeById);

        //     // Update trade
        //     console.log("Updating first trade's profit_loss...");
        //     const updatedTrade = await updateTrade(allTrades[0].id, { profit_loss: 100.0 });
        //     console.log("Updated trade", updatedTrade);

        //     // Delete trade
        //     console.log("Deleting the first trade...");
        //     const deletedTrade = await deleteTrade(allTrades[0].id);
        //     console.log("Deleted trade", deletedTrade);
        // };

        // // Test Decision Rules Helper FNs
        // console.log("Starting to test decision rules...");

        // // Get all decision rules
        // console.log("Calling getAllDecisionRules...");
        // const allDecisionRules = await getAllDecisionRules();
        // console.log("All decision rules", allDecisionRules);

        // // Assuming at least one decision rule is created successfully
        // if (allDecisionRules.length > 0) {
        //     // Get decision rule by ID
        //     console.log("Calling getDecisionRuleById for the first rule...");
        //     const decisionRuleById = await getDecisionRuleById(allDecisionRules[0].id);
        //     console.log("Decision rule by ID", decisionRuleById);

        //     // Update decision rule
        //     console.log("Updating first decision rule's value...");
        //     const updatedDecisionRule = await updateDecisionRule(allDecisionRules[0].id, { value: 'RSI < 25' });
        //     console.log("Updated decision rule", updatedDecisionRule);

        //     // Delete decision rule
        //     console.log("Deleting the first decision rule...");
        //     const deletedDecisionRule = await deleteDecisionRule(allDecisionRules[0].id);
        //     console.log("Deleted decision rule", deletedDecisionRule);
        // };

        console.log("Starting to test CPI data functions...");

        // Create a new CPI data entry
        console.log("Calling storeCPIData to create a new entry...");
        const newCPIData = await storeCPIData({ date: '2024-01-01', value: 260.474 });
        console.log("New CPI data entry", newCPIData);

        // Get all CPI data entries
        console.log("Calling getAllCPIData...");
        const allCPIData = await getAllCPIData();
        console.log("All CPI data entries", allCPIData);

        // Assuming at least one CPI data entry is created successfully
        if (allCPIData.length > 0) {
            // Get CPI data by date
            console.log("Calling getCPIDataByDate for the first entry...");
            const cpiDataByDate = await getCPIDataByDate(allCPIData[0].date);
            console.log("CPI data by date", cpiDataByDate);

            // Update CPI data
            console.log("Updating first CPI data entry's value...");
            const updatedCPIData = await updateCPIDataByDate(allCPIData[0].date, 261.474);
            console.log("Updated CPI data entry", updatedCPIData);

            // Delete CPI data
            console.log("Deleting the first CPI data entry...");
            const deletedCPIData = await deleteCPIDataByDate(allCPIData[0].date);
            console.log("Deleted CPI data entry", deletedCPIData);
        }



        // // Test Alerts Helper FNs
        // console.log("Starting to test alerts...");

        // // Get all alerts
        // console.log("Calling getAllAlerts...");
        // const allAlerts = await getAllAlerts();
        // console.log("All alerts", allAlerts);

        // // Assuming at least one alert is created successfully
        // if (allAlerts.length > 0) {
        //     // Get alert by ID
        //     console.log("Calling getAlertById for the first alert...");
        //     const alertById = await getAlertById(allAlerts[0].id);
        //     console.log("Alert by ID", alertById);

        //     // Update alert
        //     console.log("Updating first alert's message...");
        //     const updatedAlert = await updateAlert(allAlerts[0].id, { message: 'Updated message' });
        //     console.log("Updated alert", updatedAlert);

        //     // Delete alert
        //     console.log("Deleting the first alert...");
        //     const deletedAlert = await deleteAlert(allAlerts[0].id);
        //     console.log("Deleted alert", deletedAlert);
        // };

        // // Test Audit Logs Helper FNs
        // console.log("Starting to test audit logs...");

        // // Get all audit logs
        // console.log("Calling getAllAuditLogs...");
        // const allAuditLogs = await getAllAuditLogs();
        // console.log("All audit logs", allAuditLogs);

        // // Assuming at least one audit log is created successfully
        // if (allAuditLogs.length > 0) {
        //     // Get audit log by ID
        //     console.log("Calling getAuditLogById for the first log...");
        //     const auditLogById = await getAuditLogById(allAuditLogs[0].id);
        //     console.log("Audit log by ID", auditLogById);

        //     // Delete audit log
        //     console.log("Deleting the first audit log...");
        //     const deletedAuditLog = await deleteAuditLog(allAuditLogs[0].id);
        //     console.log("Deleted audit log", deletedAuditLog);
        // };

        // // Test Audit Logs Helper FNs
        // console.log("Starting to test real-time SPX data...");

        // // Get all real-time SPX records
        // console.log("Calling getAllRealTimeSPXRecords...");
        // const allRealTimeSPXRecords = await getAllRealTimeSPXRecords();
        // console.log("All real-time SPX records", allRealTimeSPXRecords);

        // // Assuming at least one real-time SPX record is created successfully
        // if (allRealTimeSPXRecords.length > 0) {
        //     // Get real-time SPX record by ID
        //     console.log("Calling getRealTimeSPXRecordById for the first record...");
        //     const realTimeSPXRecordById = await getRealTimeSPXRecordById(allRealTimeSPXRecords[0].id);
        //     console.log("Real-time SPX record by ID", realTimeSPXRecordById);

        //     // Update real-time SPX record
        //     console.log("Updating first real-time SPX record's current_price...");
        //     const updatedRealTimeSPXRecord = await updateRealTimeSPXRecord(allRealTimeSPXRecords[0].id, { current_price: 5290.0 });
        //     console.log("Updated real-time SPX record", updatedRealTimeSPXRecord);

        //     // Delete real-time SPX record
        //     console.log("Deleting the first real-time SPX record...");
        //     const deletedRealTimeSPXRecord = await deleteRealTimeSPXRecord(allRealTimeSPXRecords[0].id);
        //     console.log("Deleted real-time SPX record", deletedRealTimeSPXRecord);
        // };
    } catch (error) {
        console.log("Error during testDB!");
        console.log(error);
    }
};

// Seed and Import
async function seedAndImport() {
    try {
        await rebuildDB();
        // await importSpecificCSVFile();
        // await importDetailedHistoricalSPX();
        await testDB();
        console.log('Seed and import completed successfully.');
    } catch (error) {
        console.error("Error during seedAndImport!", error);
    } finally {
        await client.end(); 
    }
};

seedAndImport();