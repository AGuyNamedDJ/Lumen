const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const { getUserByUsername } = require('../../db/helperFunctions/user');
const JWT_SECRET = process.env.JWT_SECRET;

const LoginRouter = express.Router();

LoginRouter.post('/login', async (req, res) => {
  const { username, password } = req.body;

  try {
    const user = await getUserByUsername(username);

    if (!user) {
      return res.status(401).json({ error: 'Invalid username or password' });
    }

    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid) {
      return res.status(401).json({ error: 'Invalid username or password' });
    }

    const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, { expiresIn: '1h' });

    res.json({ token });
  } catch (error) {
    console.error('Failed to log in:', error);
    res.status(500).json({ error: 'Failed to log in' });
  }
});

module.exports = LoginRouter;