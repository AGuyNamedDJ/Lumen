# Software Development Life Cycle (SDLC) Plan

This document outlines the high-level strategy and plan for developing the backend of the Lumen quantitative trading platform.

## Table of Contents <a name="table-of-contents"></a>

1. [Inception](#inception)
2. [Design](#design)
3. [Implementation](#implementation)
4. [Testing](#testing)
5. [Deployment & Maintenance](#d-m)
   - [Deployment](#deployment)
   - [Maintenance](#maintenance)

---

## Inception Overview <a name="inception"></a>

1. **Identified the key stakeholders**:
   - The main stakeholders of this project are myself, Dalron J. Robertson, and other traders I share it with. The system is built for my personal use and for enhancing the trading strategies of others.
2. **Gathered requirements**:
   - Through my extensive experience and analysis of existing systems, I determined the essential features for the backend system. These included managing user accounts, handling market data, implementing AI models, and ensuring secure data access.
3. **Defined the system's scope**:
   - Based on the requirements, I defined the scope of the backend application to ensure it would fully support the functionalities required for quantitative trading.
4. **Outlined initial resources and timelines**:
   - I determined the resources needed, such as development tools and technologies, and created a preliminary timeline for the development process.
5. **Designed the system architecture**:
   - A crucial step was designing a comprehensive database schema to ensure efficient data management. The schema was built with a focus on scalability and optimization to handle various data such as user information, market data, AI models, and trading strategies.

---

## Design <a name="design"></a>

1. **Database Design**:
   - Using the requirements I gathered, I designed a relational database schema in PostgreSQL. The schema includes several tables to efficiently manage users, market data, AI models, and trading strategies. With careful planning and normalization, I ensured data integrity and efficient storage.
2. **API Design**:
   - Next, I designed RESTful APIs to expose the necessary endpoints for the frontend application to interact with the data. I followed best practices for API design, including the use of HTTP methods, status codes, and sensible endpoint paths.
3. **Application Architecture Design**:
   - I adopted the MVC (Model-View-Controller) architecture for the application. With Express.js as the main framework, I structured the backend into Models for database interactions, Controllers for handling business logic, and Routes for endpoint configurations.
4. **Security Design**:
   - Security is paramount. I planned for the use of bcrypt for password hashing and jsonwebtoken for user authentication. Additionally, I designed the application to follow the principle of least privilege, ensuring each user role has only the necessary access rights.
5. **Error Handling and Logging Design**:
   - To enhance maintainability and debuggability, I planned for comprehensive error handling and logging using tools like Morgan.
6. **Performance Considerations**:
   - While designing the backend, I considered the system's performance under different load conditions. I made provisions for optimizing database queries and handling potential bottlenecks in the application.

---

## Implementation <a name="implementation"></a>

During the implementation phase of the project, I translated the application design and plan into actual code. I adopted a modular approach, focusing on implementing one feature at a time, which improved the clarity of the code and facilitated easier debugging and testing. The implementation phase was divided into various stages, as detailed below:

1. **Environment Setup**:
   - I started by setting up the development environment. This involved installing the necessary software and libraries such as Node.js, Express, PostgreSQL, and others. I also initialized the project using npm, creating a package.json file to track the project's dependencies and metadata.
2. **Database Creation**:
   - Using PostgreSQL, I designed and implemented the database schema based on the models identified during the design phase. I wrote a seed script to set up and populate the database with initial data for testing purposes.
3. **Backend Implementation**:
   - I built the server using Express.js, a popular Node.js framework. I adhered to REST principles in the architecture, creating API routes corresponding to standard HTTP methods (GET, POST, PUT, DELETE). The controllers, defined in Express.js, interacted with the models to retrieve data and send it to the client side.
4. **Authentication and Authorization**:
   - I implemented user authentication using JWT (JSON Web Tokens) and Bcrypt for password hashing. This ensured secure user login and signup. I also set up middleware functions to protect certain routes and maintain user sessions.
5. **Integration**:
   - After completing the individual components, I focused on integrating all parts of the application. I ensured that the API endpoints correctly interacted with the database and returned the expected results. I also verified that the application correctly handled errors and edge cases.
6. **Performance Optimization**:
   - I optimized the application for better performance. This included improving database queries, minimizing HTTP requests, and implementing caching where appropriate.

---

## Testing <a name="testing"></a>

Testing is a critical phase in the software development lifecycle. It helps ensure the functionality, reliability, performance, and security of the application. For this project, manual testing was conducted through extensive logging and step-by-step verification of each functionality.

### Testing Approach

1. **Logging**: Throughout the codebase, `console.log` statements were used to trace the execution flow and validate the data at various stages of processing.
2. **Endpoint Verification**: Each API endpoint was manually tested using tools like Postman to ensure they work as expected. This included verifying the responses for different request types (GET, POST, PUT, DELETE).
3. **Error Handling**: Specific scenarios were tested to check how the system handles errors, such as invalid input data or unauthorized access attempts.
4. **Database Operations**: Database operations (CRUD) were verified by directly querying the PostgreSQL database before and after API calls to ensure data consistency.

### Example Testing Process

For example, to test the **Fetch Stock Price** endpoint:

1. **Logging in Code**: Add `console.log` statements in `stockPrice.js` to log the incoming request data and the response being sent back.

   ```javascript
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

2. **Manual Request with Postman**:

   - Open Postman and create a GET request to `http://localhost:3000/api/finnhubAPI/stockPrice`.
   - In the query parameters, include the stock symbol, e.g., `symbol=AAPL`.
   - Send the request and observe the response.

3. **Verify Logs**:

   - Check the server logs to ensure the data was received and processed correctly.
   - Verify the logs show the expected data at each stage of the process.

4. **Database Verification**:
   - Use a PostgreSQL client to query the relevant table and verify that the data has been updated or retrieved correctly.

By following this detailed manual testing process, you can ensure each part of the system works as intended and catch any issues early.

---

## Deployment & Maintenance <a name="d-m"></a>

### Deployment <a name="deployment"></a>

Deployment is the phase where the application is made available to end users. For the Lumen project, I've chosen [Render](https://render.com) as the deployment platform, given its simplicity, reliability, and excellent support for Node.js applications.

Render enables automatic deployments from your GitHub or GitLab repositories, along with integrated support for HTTPS, custom domains, and continuous integration/continuous deployment (CI/CD).

Here's a snapshot of the deployment process:

1. **Push to Repository**: I commit and push the finalized application code to the repository.
2. **Connect to Render**: Link the GitHub repository to the Render account. This sets up Render to watch for changes in the repository.
3. **Automatic Deployments**: Render automatically deploys the application whenever I push to the selected branch of the repository. This ensures the application is always up-to-date with the latest changes.
4. **Database Connection**: Configure the environment variables on Render to securely connect to the PostgreSQL database.
5. **Verify Deployment**: Once Render deploys the application, I thoroughly test it to ensure it functions correctly in the live environment.

### Maintenance <a name="maintenance"></a>

Maintenance is an ongoing process of monitoring, updating, and improving the application post-deployment. I use Render's integrated metrics and analytics to continually monitor the application's performance and health.

1. **Monitor Performance**: I continuously keep tabs on the application's performance, reliability, and usage patterns using Render's analytics tools.
2. **Updates and Improvements**: As I collect user feedback and data, I iterate on the application, making updates and improvements as necessary. These changes are tested in the development environment before being deployed to the live site.
3. **Security Updates**: I stay alert to any potential security vulnerabilities and promptly update the application with necessary security patches.

Through these Deployment and Maintenance procedures, I ensure that the application is not only always accessible to users but also continues to meet and exceed their needs over time. This also helps maintain a robust, secure, and high-performing application that aligns with industry best practices.

---

## Credits <a name="credits"></a>

Lumen was designed and developed by Dalron J. Robertson, showcasing his expertise in backend development, quantitative trading, and AI model training. This project reflects a commitment to creating efficient, secure, and scalable solutions for advanced trading strategies.

    - **Project Lead and Developer**: Dalron J. Robertson

---

## Contact Information <a name="contact-information"></a>

For any questions, feedback, or contributions, please contact:

- **Dalron J. Robertson**
- **Email**: dalronjrobertson@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/dalronjrobertson)
- **GitHub**: [Your GitHub Profile](https://github.com/dalronjrobertson)
