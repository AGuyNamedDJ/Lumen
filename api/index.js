require("dotenv").config();
const express = require("express");
const apiRouter = express.Router();
const jwt = require('jsonwebtoken');
const JWT_SECRET = process.env.JWT_SECRET;
const { getUserById } = require('../db/helperFunctions/user');

// JWT Middleware for authentication
apiRouter.use(async (req, res, next) => {
    const prefix = 'Bearer ';
    const auth = req.header('Authorization');

    if (!auth) {
        next();
    } else if (auth.startsWith(prefix)) {
        const token = auth.slice(prefix.length);
        try {
            const parsedToken = await jwt.verify(token, JWT_SECRET);
            const id = parsedToken && parsedToken.id;
            if (id) {
                req.user = await getUserById(id);
                next();
            }
        } catch (error) {
            next();
        }
    } else {
        next({
            name: 'AuthorizationHeaderError',
            message: `Authorization token must start with ${ prefix }`
        });
    }
});

// Handle GET request to '/api' to check if API is running
apiRouter.get("/", (req, res) => {
    res.json({ message: 'API is running' });
});

// General error handling middleware
apiRouter.use((error, req, res, next) => {
    res.status(500).send({ message: error.message });
});

// Importing and using routers
const finnhubRoutes = require('./finnhubAPI');
const LoginRouter = require('./helperFunctions/login');
const lumen1Router = require('./lumen_1');

apiRouter.use('/finnhub', finnhubRoutes);
apiRouter.use('/login', LoginRouter);
apiRouter.use('/lumen_1', lumen1Router);

module.exports = { apiRouter };