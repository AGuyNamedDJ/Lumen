const express = require('express');
const dataRouter = express.Router();
const fs = require('fs');
const path = require('path');

// Path
const dataDir = path.join(__dirname, '../../data/lumen_1/processed');

// Endpoint to upload new data
dataRouter.post('/upload', (req, res, next) => {
    try {
        const { filename, data } = req.body;
        if (!filename || !data) {
            return res.status(400).json({ success: false, message: 'Filename and data are required' });
        }
        const filePath = path.join(dataDir, filename);
        fs.writeFileSync(filePath, data);
        res.json({ success: true, message: 'File uploaded successfully' });
    } catch (error) {
        next(error);
    }
});

// Endpoint to fetch data
dataRouter.get('/fetch', (req, res, next) => {
    try {
        const { filename } = req.query;
        if (!filename) {
            return res.status(400).json({ success: false, message: 'Filename is required' });
        }
        const filePath = path.join(dataDir, filename);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ success: false, message: 'File not found' });
        }
        const data = fs.readFileSync(filePath, 'utf-8');
        res.json({ success: true, data });
    } catch (error) {
        next(error);
    }
});

// Endpoint to update existing data
dataRouter.put('/update', (req, res, next) => {
    try {
        const { filename, newData } = req.body;
        if (!filename || !newData) {
            return res.status(400).json({ success: false, message: 'Filename and new data are required' });
        }
        const filePath = path.join(dataDir, filename);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ success: false, message: 'File not found' });
        }
        fs.writeFileSync(filePath, newData);
        res.json({ success: true, message: 'File updated successfully' });
    } catch (error) {
        next(error);
    }
});

// Endpoint to delete data
dataRouter.delete('/delete', (req, res, next) => {
    try {
        const { filename } = req.body;
        if (!filename) {
            return res.status(400).json({ success: false, message: 'Filename is required' });
        }
        const filePath = path.join(dataDir, filename);
        if (!fs.existsSync(filePath)) {
            return res.status(404).json({ success: false, message: 'File not found' });
        }
        fs.unlinkSync(filePath);
        res.json({ success: true, message: 'File deleted successfully' });
    } catch (error) {
        next(error);
    }
});

module.exports = dataRouter;