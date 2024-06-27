# Lumen

## Description <a name="description"></a>

Lumen is an advanced platform designed for quantitative trading strategies, incorporating AI-driven price prediction models and comprehensive user management systems. Inspired by institutional-grade practices from entities like Citadel, Lumen supports the development, training, and deployment of machine learning models to achieve precise market forecasting.

---

## Table of Contents <a name="table-of-contents"></a>

1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File & Directory Structure](#file-directory-structure)
   - [ai/](ai)
   - [api/](#api)
   - [db/](#db)
5. [APIs & Libraries Used](#api)
6. [Testing](#testing)
7. [Credits](#credits)
8. [Contact Information](#contact-information)

---

## Installation <a name="installation"></a>

To install and set up the Lumen platform, follow these steps:

1. **Clone the repository**:

   ```sh
   git clone https://github.com/your-username/lumen.git
   cd lumen
   ```

2. **Install dependencies**:

   ```sh
   npm install
   ```

3. **Set up environment variables**:
   Create a .env file in the root directory and add your configuration settings as shown below:

```env
DATABASE_URL=your_database_url
JWT_SECRET=your_jwt_secret
FINNHUB_API_KEY=your_finnhub_api_key
```

4. **Start the development server**:

```sh
npm run dev
```

---

## Usage <a name="usage"></a>

After successfully running the server, navigate to localhost:3000 (or the port indicated in your terminal) in your browser. You should see the landing page for the Lumen platform.

---

### Key Features:

    1. AI Models: Train and deploy AI models for price prediction.
    2. User Management: Handle user registration, authentication, and authorization.
    3. Market Data: Fetch and process real-time market data from Finnhub.
    4. Trading Strategies: Implement and backtest quantitative trading strategies.

---

## File & Directory Structure <a name="file-directory-structure"></a>

The Lumen project is organized as follows:

### ai/ <a name="ai"></a>

Contains the AI models and related utilities.

    • data/: Datasets for training models.
    • models/: Machine learning models.
    • utils/: Utility functions for data preprocessing and model evaluation.

### api/ <a name="api"></a>

Contains the API endpoint definitions and logic.

    • finnhubAPI/: Endpoints for interacting with the Finnhub API.
    • companyNews.js: Fetches company news.
    • economicData.js: Retrieves economic data.
    • finnhubAPI.js: General Finnhub API interaction.
    • finnhubWebsocket.js: Manages websocket connections.
    • marketInfo.js: Gets market information.
    • stockPrice.js: Fetches stock price data.
    • index.js: Entry point for the API routes.

### db/ <a name="db"></a>

Contains the database models and helper functions.

    • csvHistory/: Handles historical CSV data.
    • helperFunctions/: Various helper functions.
    • alerts.js: Manages alerts.
    • auditLogs.js: Logs audits.
    • decisionRules.js: Decision rules for trading strategies.
    • detailedHistorical.js: Detailed historical data processing.
    • historicalSPX.js: Historical S&P 500 data.
    • realTimeSPX.js: Real-time S&P 500 data.
    • strategies.js: Trading strategies.
    • trades.js: Records of trades.
    • user.js: User management.
    • handleRealTimeData.js: Manages real-time data processing.
    • importDetailedHistoricalData.js: Imports detailed historical data.
    • importHistoricalData.js: Imports historical data.
    • index.js: Entry point for database connections and model aggregation.
    • seed.js: Script for seeding the database.

Root Directory

    • .env: Contains environment variables for configuration.
    • .gitattributes: Git attributes configuration.
    • .gitignore: Specifies files and directories to be ignored by Git.
    • package-lock.json: Contains the exact versions of dependencies installed.
    • package.json: Lists project dependencies and scripts.
    • README.md: The main documentation file for the project.
    • requirements.txt: Python dependencies for the AI models.
    • server.js: Entry point for starting the server.

---

## APIs & Libraries Used <a name="api"></a>

### Libraries:

    1. axios: A promise-based HTTP client for making API requests.
    2. bcrypt: A library for hashing passwords securely.
    3. bottleneck: A rate limiter to control the flow of API requests.
    4. cors: Middleware to enable Cross-Origin Resource Sharing.
    5. csv-parser: A module to parse CSV files.
    6. dotenv: A module to manage environment variables.
    7. express: A web framework for Node.js to build APIs and web applications.
    8. fast-csv: A CSV parsing and formatting library.
    9. finnhub: A client library to interact with the Finnhub API.
    10. jsonwebtoken: A library to handle JSON Web Tokens for authentication.
    11.	morgan: An HTTP request logger middleware.
    12.	pg: A PostgreSQL client for Node.js.
    13. validator: A library for string validation and sanitization.
    14.	ws: A websocket library for Node.js.

### APIs:

Lumen provides a comprehensive set of RESTful APIs to manage quantitative trading strategies and AI models. Here’s an overview of the key functionalities provided by the APIs:

### Authentication and User Management

Lumen includes secure user authentication and authorization using JSON Web Tokens (JWT). It ensures that only authorized personnel can access sensitive trading data.

    • Register User: POST /api/users/register - Registers a new user with the system.
    • Login User: POST /api/users/login - Authenticates a user and issues a JWT.

### Market Data Integration

Fetch and process real-time and historical market data from Finnhub.

    • Fetch Stock Price: GET /api/finnhubAPI/stockPrice - Retrieves current stock price data.
    • Fetch Company News: GET /api/finnhubAPI/companyNews - Retrieves recent news articles about a company.

### AI Model Management

Develop, train, and deploy AI models for price prediction.

    • Train Model: POST /api/ai/models/train - Trains a new AI model.
    • Deploy Model: POST /api/ai/models/deploy - Deploys a trained AI model.

### Example Usage

To get a feel for how to interact with the Lumen APIs, here’s an example of how to create a new user:

```sh
curl -X POST http://localhost:3000/api/users/register -H "Content-Type: application/json" -d '{
  "username": "johndoe",
  "password": "securepassword",
  "email": "john.doe@example.com"
}'
```

This request will create a new user in the system. Similar requests can be made for other endpoints to manage market data, AI models, and trading strategies.

---

## Testing <a name="testing"></a>

Testing is a crucial part of the development process to ensure the reliability and functionality of the Lumen platform. Manual testing was conducted through extensive logging and step-by-step verification of each functionality.

### Testing Approach

    1. Logging: Throughout the codebase, console.log statements were used to trace the execution flow and validate the data at various stages of processing.
    2. Endpoint Verification: Each API endpoint was manually tested using tools like Postman to ensure they work as expected. This included verifying the responses for different request types (GET, POST, PUT, DELETE).
    3. Error Handling: Specific scenarios were tested to check how the system handles errors, such as invalid input data or unauthorized access attempts.
    4. Database Operations: Database operations (CRUD) were verified by directly querying the PostgreSQL database before and after API calls to ensure data consistency.

### Example Testing Process

For example, to test the Fetch Stock Price endpoint:

    1. Logging in Code: Add console.log statements in stockPrice.js to log the incoming request data and the response being sent back.

```js
router.get("/stockPrice", async (req, res) => {
  try {
    console.log("Fetching stock price for:", req.query.symbol);
    const stockPrice = await getStockPrice(req.query.symbol);
    console.log("Stock price fetched:", stockPrice);
    res.status(200).json(stockPrice);
  } catch (error) {
    console.error("Error fetching stock price:", error);
    res.status(500).json({ error: "Failed to fetch stock price" });
  }
});
```

    2. Manual Request with Postman:
        • Open Postman and create a GET request to http://localhost:3000/api/finnhubAPI/stockPrice.
        • In the query parameters, include the stock symbol, e.g., symbol=AAPL.
        • Send the request and observe the response.
    3.Verify Logs:
        • Check the server logs to ensure the data was received and processed correctly.
        • Verify the logs show the expected data at each stage of the process.
    4. Database Verification:
        • Use a PostgreSQL client to query the relevant table and verify that the data has been updated or retrieved correctly.

By following this detailed manual testing process, you can ensure each part of the system works as intended and catch any issues early.

---

## Credits <a name="credits"></a>

HealthHive was designed and developed by Dalron J. Robertson, showcasing his expertise in backend development and his commitment to creating efficient, secure, and scalable solutions for healthcare data management.

    • Project Lead and Developer: Dalron J. Robertson

---

## Contact Information <a name="contact-information"></a>

- Email: dalronj.robertson@gmail.com
- Github: [AGuyNamedDJ](https://github.com/AGuyNamedDJ)
- LinkedIn: [Dalron J. Robertson](https://www.linkedin.com/in/dalronjrobertson/)
- Website: [dalronjrobertson.com](https://dalronjrobertson.com)
- YouTube: [AGNDJ](https://youtube.com/@AGNDJ)

I'm always open to feedback, collaboration, or simply a chat. Feel free to get in touch!
