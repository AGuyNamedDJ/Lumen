const express = require('express');
const jwt = require('jsonwebtoken');
const { createUser, getAllUsers, getUserById, getUserByUsername, deleteUser, updateUser } = require('../../db/helperFunctions/user');
const JWT_SECRET = process.env.JWT_SECRET;

console.log('JWT_SECRET during token verification:', JWT_SECRET);

const signupRouter = express.Router();

signupRouter.post('/', async (req, res) => {
    const { email, username, password, first_name, last_name } = req.body;

    console.log('Received signup request:');
    console.log('Email:', email);
    console.log('Username:', username);
    console.log('Password:', password);
    console.log('First Name:', first_name);
    console.log('Last Name:', last_name);

    try {
        console.log('Attempting to create user...');
        const user = await createUser({ email, username, password, first_name, last_name });

        if (user) {
            console.log('User created successfully:', user);
            res.status(201).json({ message: 'User created successfully', user });
        } else {
            console.log('Username already exists.');
            res.status(400).json({ message: 'Username already exists' });
        }
    } catch (error) {
        console.error('Error creating user:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

module.exports = signupRouter;