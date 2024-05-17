// Import Client & Exports;
const { client } = require('./index');

// File Imports
const importSpecificCSVFile = require('./importHistoricalData');
const { createUser, getAllUsers, getUserById, getUserByUsername, deleteUser, updateUser, loginUser } = require('./helperFunctions/user');
const { createStrategy, getAllStrategies, getStrategyByName, getStrategyById, updateStrategy, deleteStrategy } = require('./helperFunctions/strategies');



// Methods: Drop Tables
async function dropTables(){
    try {
        console.log("Dropping tables... ");
        await client.query(`
            DROP TABLE IF EXISTS users CASCADE;
            DROP TABLE IF EXISTS strategies CASCADE;
            DROP TABLE IF EXISTS trades CASCADE;
            DROP TABLE IF EXISTS decision_rules;
            DROP TABLE IF EXISTS alerts CASCADE;
            DROP TABLE IF EXISTS market_data CASCADE;
            DROP TABLE IF EXISTS audit_logs CASCADE;
            DROP TABLE IF EXISTS configurations CASCADE;
            DROP TABLE IF EXISTS historical_spx CASCADE;
            DROP TABLE IF EXISTS real_time_spx CASCADE;
        `);
        console.log("Finished dropping tables.")
    } catch(error){
        console.log("Error dropping tables!")
        console.log(error)
    }
};

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
        CREATE TABLE IF NOT EXISTS historical_spx (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume BIGINT
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

        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open_price NUMERIC NOT NULL,
            close_price NUMERIC NOT NULL,
            high_price NUMERIC NOT NULL,
            low_price NUMERIC NOT NULL,
            volume BIGINT
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            action_type VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS configurations (
            id SERIAL PRIMARY KEY,
            key VARCHAR(255) UNIQUE NOT NULL,
            value TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS real_time_spx (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume BIGINT
        );
    `);
    console.log('Finished building tables.');
    } catch (error) {
        console.error('Error building tables!');
        console.log(error);
    }
};

// createInitialUsers
async function createInitialUsers() {
    console.log("Creating initial users...");
    try {
        await createUser({
            username: 'Owner1', 
            password: 'SecurePass123!', 
            email: 'user1@example.com', 
            first_name: 'Dalron', 
            last_name: 'Robertson', 
            phone_number: '601-456-7890',
            date_of_birth: '1980-01-01',
            role: 'admin'
        });

        console.log("Finished creating initial users.");
    } catch (error) {
        console.error("Error creating initial users!");
        console.error(error);
    }
};

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


// Rebuild DB
async function rebuildDB() {
    try {
        await client.connect();
        await dropTables();
        await createTables();
        await createInitialUsers();
        await createInitialStrategies();
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

        // Test Strategies Helper FNs
        console.log("Starting to test strategies...");
        await createInitialStrategies();

        // Get all strategies
        console.log("Calling getAllStrategies...");
        const allStrategies = await getAllStrategies();
        console.log("All strategies", allStrategies);

        // Assuming at least one strategy is created successfully
        if (allStrategies.length > 0) {
            // Get strategy by ID
            console.log("Calling getStrategyById for the first strategy...");
            const strategyById = await getStrategyById(allStrategies[0].id);
            console.log("Strategy by ID", strategyById);

            // Update strategy
            console.log("Updating first strategy's description...");
            const updatedStrategy = await updateStrategy(allStrategies[0].id, { description: 'Updated strategy description' });
            console.log("Updated strategy", updatedStrategy);

            // Delete strategy
            console.log("Deleting the first strategy...");
            const deletedStrategy = await deleteStrategy(allStrategies[0].id);
            console.log("Deleted strategy", deletedStrategy);
        }


    } catch (error) {
        console.log("Error during testDB!");
        console.log(error);
    }
};




// Seed and Import
async function seedAndImport() {
    try {
        await rebuildDB();
        await importSpecificCSVFile(); // Import historical data after rebuilding tables
        await testDB();
        console.log('Seed and import completed successfully.');
    } catch (error) {
        console.error("Error during seedAndImport!", error);
    } finally {
        await client.end(); 
    }
};

seedAndImport();